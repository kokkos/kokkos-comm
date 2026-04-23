// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <type_traits>

#include <gtest/gtest.h>
#include <KokkosComm/KokkosComm.hpp>

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/utils.hpp"
#endif

namespace {

// from_raw
// --------

TEST(Communicator, from_raw_returns_same_communicator) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto result = KokkosComm::Communicator<>::from_raw(raw_comm);

  ASSERT_EQ(result.comm(), raw_comm);
}

TEST(Communicator, from_raw_preserves_size_and_rank) {
  int expected_size, expected_rank;
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
  ncclCommCount(raw_comm, &expected_size);
  ncclCommUserRank(raw_comm, &expected_rank);
#else
  auto raw_comm = MPI_COMM_WORLD;
  MPI_Comm_size(raw_comm, &expected_size);
  MPI_Comm_rank(raw_comm, &expected_rank);
#endif
  auto comm = KokkosComm::Communicator<>::from_raw(raw_comm);

  ASSERT_EQ(comm.size(), expected_size);
  ASSERT_EQ(comm.rank(), expected_rank);
}

TEST(Communicator, exec_returns_execution_space_of_correct_type) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  using Exec = Kokkos::DefaultExecutionSpace;
  auto comm  = KokkosComm::Communicator<>::from_raw(raw_comm);

  static_assert(std::is_same_v<decltype(comm.exec()), const Exec&>);
  ASSERT_TRUE(&comm.exec() != nullptr);
}

// duplicate
// ---------

TEST(Communicator, duplicate_from_raw_returns_valid_communicator) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto result = KokkosComm::Communicator<>::duplicate_from_raw(raw_comm);

  ASSERT_TRUE(result.has_value());
}

TEST(Communicator, duplicate_from_raw_preserves_size_and_rank) {
  int expected_size, expected_rank;
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
  ncclCommCount(raw_comm, &expected_size);
  ncclCommUserRank(raw_comm, &expected_rank);
#else
  auto raw_comm = MPI_COMM_WORLD;
  MPI_Comm_size(raw_comm, &expected_size);
  MPI_Comm_rank(raw_comm, &expected_rank);
#endif
  auto result = KokkosComm::Communicator<>::duplicate_from_raw(raw_comm).value();

  ASSERT_EQ(result.size(), expected_size);
  ASSERT_EQ(result.rank(), expected_rank);
}

TEST(Communicator, duplicate_preserves_size_and_rank) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto original = KokkosComm::Communicator<>::from_raw(raw_comm);
  auto result   = original.duplicate().value();

  ASSERT_EQ(result.size(), original.size());
  ASSERT_EQ(result.rank(), original.rank());
}

TEST(Communicator, duplicate_produces_independent_communicator) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto original = KokkosComm::Communicator<>::from_raw(raw_comm);
  auto result   = original.duplicate().value();

  ASSERT_NE(result.comm(), original.comm());
}

TEST(Communicator, duplicate_chain_produces_independent_communicators) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto comm1 = KokkosComm::Communicator<>::from_raw(raw_comm);
  auto comm2 = comm1.duplicate().value();
  auto comm3 = comm2.duplicate().value();

  ASSERT_NE(comm1.comm(), comm2.comm());
  ASSERT_NE(comm2.comm(), comm3.comm());
  ASSERT_NE(comm1.comm(), comm3.comm());
}

// split
// -----

TEST(Communicator, split_from_raw_returns_valid_communicator) {
  int rank;
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
  ncclCommUserRank(raw_comm, &rank);
#else
  auto raw_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(raw_comm, &rank);
#endif
  auto result = KokkosComm::Communicator<>::split_from_raw(raw_comm, 0, rank);

  ASSERT_TRUE(result.has_value());
}

TEST(Communicator, split_undefined_color_returns_nullopt) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
  int color     = NCCL_SPLIT_NOCOLOR;
#else
  auto raw_comm = MPI_COMM_WORLD;
  int color     = MPI_UNDEFINED;
#endif
  auto original = KokkosComm::Communicator<>::from_raw(raw_comm);
  auto result   = original.split(color, 0);

  ASSERT_FALSE(result.has_value());
}

TEST(Communicator, split_same_color_groups_all_ranks) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto original   = KokkosComm::Communicator<>::from_raw(raw_comm);
  const int color = 0;

  auto result = original.split(color, original.rank()).value();

  ASSERT_EQ(result.size(), original.size());
}

TEST(Communicator, split_two_colors_produces_half_sized_communicators) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto original = KokkosComm::Communicator<>::from_raw(raw_comm);
  if (original.size() < 2 or original.size() % 2 != 0) {
    GTEST_SKIP() << "Requires an even number of ranks >= 2 (" << original.size() << " provided)";
  }
  const int color = original.rank() % 2;

  auto result = original.split(color, original.rank()).value();

  ASSERT_EQ(result.size(), original.size() / 2);
}

TEST(Communicator, split_key_controls_rank_ordering) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto original = KokkosComm::Communicator<>::from_raw(raw_comm);
  if (original.size() < 2 or original.size() % 2 != 0) {
    GTEST_SKIP() << "Requires an even number of ranks >= 2 (" << original.size() << " provided)";
  }
  const int color = 0;
  // Reverse key: rank 0 gets the highest key, rank N-1 gets 0
  const int reversed_key = original.size() - 1 - original.rank();

  auto result = original.split(0, reversed_key).value();

  ASSERT_EQ(result.rank(), reversed_key);
}

TEST(Communicator, sequential_splits_produce_independent_communicators) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto original = KokkosComm::Communicator<>::from_raw(raw_comm);
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

// move semantics
// --------------

TEST(Communicator, move_constructed_communicator_is_valid) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto original           = KokkosComm::Communicator<>::from_raw(raw_comm);
  const int expected_size = original.size();
  const int expected_rank = original.rank();

  auto result = std::move(original);

  ASSERT_EQ(result.size(), expected_size);
  ASSERT_EQ(result.rank(), expected_rank);
}

TEST(Communicator, move_assigned_communicator_is_valid) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto original           = KokkosComm::Communicator<>::from_raw(raw_comm);
  auto target             = original.duplicate().value();
  const int expected_size = original.size();
  const int expected_rank = original.rank();

  target = std::move(original);

  ASSERT_EQ(target.size(), expected_size);
  ASSERT_EQ(target.rank(), expected_rank);
}

TEST(Communicator, self_move_assignment_is_safe) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto comm               = KokkosComm::Communicator<>::from_raw(raw_comm);
  const int expected_size = comm.size();
  const int expected_rank = comm.rank();

  comm = std::move(comm);

  ASSERT_EQ(comm.size(), expected_size);
  ASSERT_EQ(comm.rank(), expected_rank);
}

}  // namespace
