// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include <KokkosComm/KokkosComm.hpp>

#include "view_utils.hpp"
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/utils.hpp"
#endif

namespace {

template <KokkosComm::MutKokkosView SendV, KokkosComm::MutKokkosView RecvV>
auto test_core_exchange(SendV& sv, RecvV& rv) -> void {
  using Scalar = typename RecvV::non_const_value_type;
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto& nccl_ctx = test_utils::NcclCtx::get();
  auto raw_comm  = nccl_ctx.comm();
  auto exec      = Kokkos::Cuda(nccl_ctx.stream());
#else
  auto raw_comm = MPI_COMM_WORLD;
  auto exec     = Kokkos::DefaultExecutionSpace{};
#endif
  auto comm = KokkosComm::Communicator<>::from_raw(raw_comm, exec);
  if (comm.size() < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << comm.size() << " provided)";
  }
  const int dst = (comm.rank() + 1) % comm.size();
  const int src = (comm.rank() + comm.size() - 1) % comm.size();

  Kokkos::deep_copy(exec, rv, Scalar(-1));
  test_utils::init_view(exec, sv);

  typename SendV::const_type const_sv = sv;
  auto result                         = rv;
  auto req                            = KokkosComm::exchange(comm, const_sv, dst, rv, src);
  // Clear the caller's views so only the request retains the send allocation.
  const_sv = {};
  sv       = {};
  rv       = {};
  req.wait();
  EXPECT_EQ(test_utils::count_errors(result), 0);
}

template <typename T>
class Exchange : public testing::Test {
 public:
  using Scalar = T;
};

#if defined(KOKKOSCOMM_ENABLE_NCCL)
using ScalarTypes = testing::Types<float, double, int, int64_t>;
#else
using ScalarTypes =
    testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
#endif
TYPED_TEST_SUITE(Exchange, ScalarTypes);

TYPED_TEST(Exchange, Contiguous1D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::Contig{}, "sv", 1013);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::Contig{}, "rv", 1013);
  ASSERT_TRUE(KokkosComm::is_contiguous(sv));
  ASSERT_TRUE(KokkosComm::is_contiguous(rv));
  test_core_exchange(sv, rv);
}
TYPED_TEST(Exchange, PackedSend1D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::NonContig{}, "sv", 1013);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::Contig{}, "rv", 1013);
  ASSERT_FALSE(KokkosComm::is_contiguous(sv));
  ASSERT_TRUE(KokkosComm::is_contiguous(rv));
  test_core_exchange(sv, rv);
}
TYPED_TEST(Exchange, PackedRecv1D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::Contig{}, "sv", 1013);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::NonContig{}, "rv", 1013);
  ASSERT_TRUE(KokkosComm::is_contiguous(sv));
  ASSERT_FALSE(KokkosComm::is_contiguous(rv));
  test_core_exchange(sv, rv);
}
TYPED_TEST(Exchange, PackedBoth1D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::NonContig{}, "sv", 1013);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 1>(test_utils::NonContig{}, "rv", 1013);
  ASSERT_FALSE(KokkosComm::is_contiguous(sv));
  ASSERT_FALSE(KokkosComm::is_contiguous(rv));
  test_core_exchange(sv, rv);
}
TYPED_TEST(Exchange, Contiguous2D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::Contig{}, "sv", 137, 17);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::Contig{}, "rv", 137, 17);
  ASSERT_TRUE(KokkosComm::is_contiguous(sv));
  ASSERT_TRUE(KokkosComm::is_contiguous(rv));
  test_core_exchange(sv, rv);
}
TYPED_TEST(Exchange, PackedSend2D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::NonContig{}, "sv", 137, 17);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::Contig{}, "rv", 137, 17);
  ASSERT_FALSE(KokkosComm::is_contiguous(sv));
  ASSERT_TRUE(KokkosComm::is_contiguous(rv));
  test_core_exchange(sv, rv);
}
TYPED_TEST(Exchange, PackedRecv2D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::Contig{}, "sv", 137, 17);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::NonContig{}, "rv", 137, 17);
  ASSERT_TRUE(KokkosComm::is_contiguous(sv));
  ASSERT_FALSE(KokkosComm::is_contiguous(rv));
  test_core_exchange(sv, rv);
}
TYPED_TEST(Exchange, PackedBoth2D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::NonContig{}, "sv", 137, 17);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 2>(test_utils::NonContig{}, "rv", 137, 17);
  ASSERT_FALSE(KokkosComm::is_contiguous(sv));
  ASSERT_FALSE(KokkosComm::is_contiguous(rv));
  test_core_exchange(sv, rv);
}
TYPED_TEST(Exchange, Contiguous3D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 3>(test_utils::Contig{}, "sv", 23, 17, 7);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 3>(test_utils::Contig{}, "rv", 23, 17, 7);
  ASSERT_TRUE(KokkosComm::is_contiguous(sv));
  ASSERT_TRUE(KokkosComm::is_contiguous(rv));
  test_core_exchange(sv, rv);
}
TYPED_TEST(Exchange, PackedSend3D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 3>(test_utils::NonContig{}, "sv", 23, 17, 7);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 3>(test_utils::Contig{}, "rv", 23, 17, 7);
  ASSERT_FALSE(KokkosComm::is_contiguous(sv));
  ASSERT_TRUE(KokkosComm::is_contiguous(rv));
  test_core_exchange(sv, rv);
}
TYPED_TEST(Exchange, PackedRecv3D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 3>(test_utils::Contig{}, "sv", 23, 17, 7);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 3>(test_utils::NonContig{}, "rv", 23, 17, 7);
  ASSERT_TRUE(KokkosComm::is_contiguous(sv));
  ASSERT_FALSE(KokkosComm::is_contiguous(rv));
  test_core_exchange(sv, rv);
}
TYPED_TEST(Exchange, PackedBoth3D) {
  auto sv = test_utils::build_view<typename TestFixture::Scalar, 3>(test_utils::NonContig{}, "sv", 23, 17, 7);
  auto rv = test_utils::build_view<typename TestFixture::Scalar, 3>(test_utils::NonContig{}, "rv", 23, 17, 7);
  ASSERT_FALSE(KokkosComm::is_contiguous(sv));
  ASSERT_FALSE(KokkosComm::is_contiguous(rv));
  test_core_exchange(sv, rv);
}

}  // namespace
