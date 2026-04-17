// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <type_traits>

#include <gtest/gtest.h>
#include <KokkosComm/KokkosComm.hpp>
#if defined(KOKKOSCOMM_ENABLE_MPI)
#include <mpi.h>
#elif defined(KOKKOSCOMM_ENABLE_NCCL)
#include <nccl.h>
#endif

namespace {

template <typename T>
class ReductionOperatorConversion : public testing::Test {
 public:
  using RedOp = T;
};

#if defined(KOKKOSCOMM_ENABLE_NCCL)
using RedOpTypes =
    ::testing::Types<KokkosComm::Sum, KokkosComm::Prod, KokkosComm::Min, KokkosComm::Max, KokkosComm::Average>;
#else
using RedOpTypes = ::testing::Types<
    KokkosComm::Sum,
    KokkosComm::Prod,
    KokkosComm::Min,
    KokkosComm::Max,
    KokkosComm::MinLoc,
    KokkosComm::MaxLoc,
    KokkosComm::BAnd,
    KokkosComm::LAnd,
    KokkosComm::BOr,
    KokkosComm::LOr,
    KokkosComm::BXor,
    KokkosComm::LXor>;
#endif
TYPED_TEST_SUITE(ReductionOperatorConversion, RedOpTypes);

template <KokkosComm::ReductionOperator RO>
auto test_red_op_conversion() -> void {
  using DCS = KokkosComm::DefaultCommunicationSpace;

  auto red_op = KokkosComm::reduction_op<DCS, RO>();
#if defined(KOKKOSCOMM_ENABLE_MPI)
  if constexpr (std::is_same_v<RO, KokkosComm::BAnd>) {
    ASSERT_EQ(MPI_BAND, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::BOr>) {
    ASSERT_EQ(MPI_BOR, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::BXor>) {
    ASSERT_EQ(MPI_BXOR, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::LAnd>) {
    ASSERT_EQ(MPI_LAND, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::LOr>) {
    ASSERT_EQ(MPI_LOR, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::LXor>) {
    ASSERT_EQ(MPI_LXOR, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::Max>) {
    ASSERT_EQ(MPI_MAX, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::MaxLoc>) {
    ASSERT_EQ(MPI_MAXLOC, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::Min>) {
    ASSERT_EQ(MPI_MIN, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::MinLoc>) {
    ASSERT_EQ(MPI_MINLOC, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::Sum>) {
    ASSERT_EQ(MPI_SUM, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::Prod>) {
    ASSERT_EQ(MPI_PROD, red_op);
  }
#elif defined(KOKKOSCOMM_ENABLE_NCCL)
  if constexpr (std::is_same_v<RO, KokkosComm::Sum>) {
    ASSERT_EQ(ncclSum, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::Prod>) {
    ASSERT_EQ(ncclProd, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::Min>) {
    ASSERT_EQ(ncclMin, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::Max>) {
    ASSERT_EQ(ncclMax, red_op);
  } else if constexpr (std::is_same_v<RO, KokkosComm::Average>) {
    ASSERT_EQ(ncclAvg, red_op);
  }
#else
  GTEST_SKIP() << "Unimplemented test for Default Communication Space";
#endif
}

TYPED_TEST(ReductionOperatorConversion, RedOpConv) { test_red_op_conversion<typename TestFixture::RedOp>(); }

}  // namespace
