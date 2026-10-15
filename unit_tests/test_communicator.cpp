// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include <KokkosComm/KokkosComm.hpp>

#include "communication_space_utils.hpp"

namespace {

template <typename T>
class CommunicatorTest : public testing::Test {
 public:
  using CommSpace = T;
};

using CommSpaces = ::testing::Types<
#if defined(KOKKOSCOMM_ENABLE_MPI)
    KokkosComm::MpiSpace
#if defined(KOKKOSCOMM_ENABLE_NCCL) || defined(KOKKOSCOMM_ENABLE_RCCL)
    ,
#endif
#endif
#if defined(KOKKOSCOMM_ENABLE_NCCL)
    KokkosComm::Experimental::NcclSpace
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
    KokkosComm::Experimental::RcclSpace
#endif
    >;
TYPED_TEST_SUITE(CommunicatorTest, CommSpaces);

// The default template argument is intentionally checked once, alongside the explicit-space suite.
TEST(Communicator, default_space_can_construct_from_raw) {
  using CommSpace = KokkosComm::DefaultCommunicationSpace;
  auto raw_comm   = test_utils::get_raw_comm<CommSpace>();
  auto comm       = KokkosComm::Communicator<>::from_raw(raw_comm, test_utils::get_exec<CommSpace>());

  ASSERT_EQ(comm.comm(), raw_comm);
}

TYPED_TEST(CommunicatorTest, from_raw_returns_same_communicator) {
  using CommSpace = typename TestFixture::CommSpace;
  auto raw_comm   = test_utils::get_raw_comm<CommSpace>();
  auto comm       = KokkosComm::Communicator<CommSpace>::from_raw(raw_comm, test_utils::get_exec<CommSpace>());

  ASSERT_EQ(comm.comm(), raw_comm);
}

TYPED_TEST(CommunicatorTest, from_raw_preserves_size_and_rank) {
  using CommSpace = typename TestFixture::CommSpace;
  auto raw_comm   = test_utils::get_raw_comm<CommSpace>();
  auto comm       = test_utils::make_communicator<CommSpace>();

  ASSERT_EQ(comm.size(), test_utils::get_comm_size<CommSpace>(raw_comm));
  ASSERT_EQ(comm.rank(), test_utils::get_comm_rank<CommSpace>(raw_comm));
}

TYPED_TEST(CommunicatorTest, exec_returns_execution_space_of_correct_type) {
  using Exec      = Kokkos::DefaultExecutionSpace;
  using CommSpace = typename TestFixture::CommSpace;
  auto comm       = test_utils::make_communicator<CommSpace>();

  static_assert(std::is_same_v<decltype(comm.exec()), const Exec&>);
  ASSERT_TRUE(&comm.exec() != nullptr);
}

TYPED_TEST(CommunicatorTest, duplicate_from_raw_returns_valid_communicator) {
  using CommSpace = typename TestFixture::CommSpace;
  auto raw_comm   = test_utils::get_raw_comm<CommSpace>();
  auto result = KokkosComm::Communicator<CommSpace>::duplicate_from_raw(raw_comm, test_utils::get_exec<CommSpace>());

  ASSERT_TRUE(result.has_value());
}

TYPED_TEST(CommunicatorTest, duplicate_from_raw_preserves_size_and_rank) {
  using CommSpace = typename TestFixture::CommSpace;
  auto raw_comm   = test_utils::get_raw_comm<CommSpace>();
  auto result =
      KokkosComm::Communicator<CommSpace>::duplicate_from_raw(raw_comm, test_utils::get_exec<CommSpace>()).value();

  ASSERT_EQ(result.size(), test_utils::get_comm_size<CommSpace>(raw_comm));
  ASSERT_EQ(result.rank(), test_utils::get_comm_rank<CommSpace>(raw_comm));
}

TYPED_TEST(CommunicatorTest, duplicate_preserves_size_and_rank) {
  using CommSpace = typename TestFixture::CommSpace;
  auto original   = test_utils::make_communicator<CommSpace>();
  auto result     = original.duplicate().value();

  ASSERT_EQ(result.size(), original.size());
  ASSERT_EQ(result.rank(), original.rank());
}

TYPED_TEST(CommunicatorTest, duplicate_produces_independent_communicator) {
  using CommSpace = typename TestFixture::CommSpace;
  auto original   = test_utils::make_communicator<CommSpace>();
  auto result     = original.duplicate().value();

  ASSERT_NE(result.comm(), original.comm());
}

TYPED_TEST(CommunicatorTest, duplicate_chain_produces_independent_communicators) {
  using CommSpace = typename TestFixture::CommSpace;
  auto comm1      = test_utils::make_communicator<CommSpace>();
  auto comm2      = comm1.duplicate().value();
  auto comm3      = comm2.duplicate().value();

  ASSERT_NE(comm1.comm(), comm2.comm());
  ASSERT_NE(comm2.comm(), comm3.comm());
  ASSERT_NE(comm1.comm(), comm3.comm());
}

TYPED_TEST(CommunicatorTest, split_from_raw_returns_valid_communicator) {
  using CommSpace = typename TestFixture::CommSpace;
  auto raw_comm   = test_utils::get_raw_comm<CommSpace>();
  auto exec       = test_utils::get_exec<CommSpace>();
  int rank        = test_utils::get_comm_rank<CommSpace>(raw_comm);
  auto result     = KokkosComm::Communicator<CommSpace>::split_from_raw(raw_comm, 0, rank, exec);

  ASSERT_TRUE(result.has_value());
}

TYPED_TEST(CommunicatorTest, split_undefined_color_returns_nullopt) {
  using CommSpace = typename TestFixture::CommSpace;
  auto original   = test_utils::make_communicator<CommSpace>();
  const int color = test_utils::get_undefined_color<CommSpace>();

  auto result = original.split(color, 0);

  ASSERT_FALSE(result.has_value());
}

TYPED_TEST(CommunicatorTest, split_same_color_groups_all_ranks) {
  using CommSpace = typename TestFixture::CommSpace;
  auto original   = test_utils::make_communicator<CommSpace>();
  auto result     = original.split(0, original.rank()).value();

  ASSERT_EQ(result.size(), original.size());
}

TYPED_TEST(CommunicatorTest, split_two_colors_produces_half_sized_communicators) {
  using CommSpace = typename TestFixture::CommSpace;
  auto original   = test_utils::make_communicator<CommSpace>();
  if (original.size() < 2 or original.size() % 2 != 0) {
    GTEST_SKIP() << "Requires an even number of ranks >= 2 (" << original.size() << " provided)";
  }
  const int color = original.rank() % 2;

  auto result = original.split(color, original.rank()).value();

  ASSERT_EQ(result.size(), original.size() / 2);
}

TYPED_TEST(CommunicatorTest, split_key_controls_rank_ordering) {
  using CommSpace = typename TestFixture::CommSpace;
  auto original   = test_utils::make_communicator<CommSpace>();
  if (original.size() < 2 or original.size() % 2 != 0) {
    GTEST_SKIP() << "Requires an even number of ranks >= 2 (" << original.size() << " provided)";
  }
  // Reverse key: rank 0 gets the highest key, rank N-1 gets 0.
  const int reversed_key = original.size() - 1 - original.rank();

  auto result = original.split(0, reversed_key).value();

  ASSERT_EQ(result.rank(), reversed_key);
}

TYPED_TEST(CommunicatorTest, sequential_splits_produce_independent_communicators) {
  using CommSpace = typename TestFixture::CommSpace;
  auto original   = test_utils::make_communicator<CommSpace>();
  if (original.size() < 2) {
    GTEST_SKIP() << "Requires at least 2 ranks";
  }

  const int color   = original.rank() % 2;
  auto first_split  = original.split(color, original.rank()).value();
  auto second_split = first_split.split(0, first_split.rank()).value();

  ASSERT_NE(original.comm(), first_split.comm());
  ASSERT_NE(first_split.comm(), second_split.comm());
  ASSERT_EQ(second_split.size(), first_split.size());
}

TYPED_TEST(CommunicatorTest, move_constructed_communicator_is_valid) {
  using CommSpace         = typename TestFixture::CommSpace;
  auto original           = test_utils::make_communicator<CommSpace>();
  const int expected_size = original.size();
  const int expected_rank = original.rank();

  auto result = std::move(original);

  ASSERT_EQ(result.size(), expected_size);
  ASSERT_EQ(result.rank(), expected_rank);
}

TYPED_TEST(CommunicatorTest, move_assigned_communicator_is_valid) {
  using CommSpace         = typename TestFixture::CommSpace;
  auto original           = test_utils::make_communicator<CommSpace>();
  auto target             = original.duplicate().value();
  const int expected_size = original.size();
  const int expected_rank = original.rank();

  target = std::move(original);

  ASSERT_EQ(target.size(), expected_size);
  ASSERT_EQ(target.rank(), expected_rank);
}

TYPED_TEST(CommunicatorTest, self_move_assignment_is_safe) {
  using CommSpace         = typename TestFixture::CommSpace;
  auto comm               = test_utils::make_communicator<CommSpace>();
  const int expected_size = comm.size();
  const int expected_rank = comm.rank();

  comm = std::move(comm);

  ASSERT_EQ(comm.size(), expected_size);
  ASSERT_EQ(comm.rank(), expected_rank);
}

}  // namespace
