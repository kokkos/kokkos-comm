// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <type_traits>

#include <gtest/gtest.h>
#include <KokkosComm/KokkosComm.hpp>

#if defined(KOKKOSCOMM_ENABLE_MPI)
#include <mpi.h>
#endif
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <nccl.h>
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
#include <rccl/rccl.h>
#endif

namespace {

template <typename T>
class ReductionOperatorConversionFixture : public testing::Test {
 public:
  using RedOp = T;
};

#if defined(KOKKOSCOMM_ENABLE_MPI)
using MpiRedOpTypes = ::testing::Types<
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
#if defined(KOKKOSCOMM_ENABLE_NCCL) || defined(KOKKOSCOMM_ENABLE_RCCL)
using XcclRedOpTypes =
    ::testing::Types<KokkosComm::Sum, KokkosComm::Prod, KokkosComm::Min, KokkosComm::Max, KokkosComm::Average>;
#endif

template <KokkosComm::CommunicationSpace CommSpace, KokkosComm::ReductionOperator RO>
auto test_red_op_conversion() -> void {
  auto red_op = KokkosComm::reduction_op<CommSpace, RO>();
#if defined(KOKKOSCOMM_ENABLE_MPI)
  if constexpr (std::is_same_v<CommSpace, KokkosComm::MpiSpace>) {
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
  }
#endif
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  if constexpr (std::is_same_v<CommSpace, KokkosComm::Experimental::NcclSpace>) {
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
  }
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
  if constexpr (std::is_same_v<CommSpace, KokkosComm::Experimental::RcclSpace>) {
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
  }
#endif
}

#define KC_DEFINE_RED_OP_TESTS(SuiteName, TypeList, CommSpace)       \
  template <typename T>                                              \
  class SuiteName : public ReductionOperatorConversionFixture<T> {}; \
                                                                     \
  TYPED_TEST_SUITE(SuiteName, TypeList);                             \
                                                                     \
  TYPED_TEST(SuiteName, RedOpConv) { test_red_op_conversion<CommSpace, typename TestFixture::RedOp>(); }

#if defined(KOKKOSCOMM_ENABLE_MPI)
KC_DEFINE_RED_OP_TESTS(MpiReductionOperatorConversion, MpiRedOpTypes, KokkosComm::MpiSpace)
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL)
KC_DEFINE_RED_OP_TESTS(NcclReductionOperatorConversion, XcclRedOpTypes, KokkosComm::Experimental::NcclSpace)
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
KC_DEFINE_RED_OP_TESTS(RcclReductionOperatorConversion, XcclRedOpTypes, KokkosComm::Experimental::RcclSpace)
#endif

#undef KC_DEFINE_RED_OP_TESTS

}  // namespace
