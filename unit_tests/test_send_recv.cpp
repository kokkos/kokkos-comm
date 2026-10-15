// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <KokkosComm/KokkosComm.hpp>

#include "view_utils.hpp"
#if defined(KOKKOSCOMM_ENABLE_NCCL) || defined(KOKKOSCOMM_ENABLE_RCCL)
#include "nccl/utils.hpp"
#endif

namespace {

auto test_exec() -> Kokkos::DefaultExecutionSpace {
#if defined(KOKKOSCOMM_ENABLE_NCCL) || defined(KOKKOSCOMM_ENABLE_RCCL)
  return Kokkos::DefaultExecutionSpace{test_utils::XcclCtx::get().stream()};
#else
  return Kokkos::DefaultExecutionSpace{};
#endif
}

template <KokkosComm::KokkosView View>
void test_core_send_recv(const View& v) {
#if defined(KOKKOSCOMM_ENABLE_NCCL) || defined(KOKKOSCOMM_ENABLE_RCCL)
  auto& xccl_ctx = test_utils::XcclCtx::get();
  auto raw_comm  = xccl_ctx.comm();
  auto exec      = Kokkos::DefaultExecutionSpace{xccl_ctx.stream()};
#else
  auto raw_comm = MPI_COMM_WORLD;
  auto exec     = Kokkos::DefaultExecutionSpace{};
#endif
  auto comm      = KokkosComm::Communicator<>::from_raw(raw_comm, exec);
  const int size = comm.size();
  const int rank = comm.rank();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  const int src = 0;
  const int dst = 1;

  if (rank == src) {
    test_utils::init_view(exec, v);
    exec.fence();
#if defined(KOKKOSCOMM_ENABLE_RCCL)
    int active_device = -1;
    KC_HIP_CHECK(hipGetDevice(&active_device));

    hipPointerAttribute_t attr{};
    KC_HIP_CHECK(hipPointerGetAttributes(&attr, KokkosComm::data_handle(v)));
    std::fprintf(
        stderr, "rank %d: RCCL device=%d, active HIP device=%d, view device=%d\n", rank, xccl_ctx.device(),
        active_device, attr.device
    );
#endif
    KokkosComm::send(comm, v, dst).wait();
  } else if (rank == dst) {
#if defined(KOKKOSCOMM_ENABLE_RCCL)
    int active_device = -1;
    KC_HIP_CHECK(hipGetDevice(&active_device));

    hipPointerAttribute_t attr{};
    KC_HIP_CHECK(hipPointerGetAttributes(&attr, KokkosComm::data_handle(v)));
    std::fprintf(
        stderr, "rank %d: RCCL device=%d, active HIP device=%d, view device=%d\n", rank, xccl_ctx.device(),
        active_device, attr.device
    );
#endif
    KokkosComm::recv(comm, v, src).wait();
    int errs = test_utils::count_errors(v);
    ASSERT_EQ(errs, 0);
  }
}

template <typename T>
class SendRecv : public ::testing::Test {
 public:
  using Scalar = T;
};

#if defined(KOKKOSCOMM_ENABLE_NCCL) || defined(KOKKOSCOMM_ENABLE_RCCL)
using ScalarTypes = ::testing::Types<float, double, int, int64_t>;
#else
using ScalarTypes =
    ::testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
#endif
TYPED_TEST_SUITE(SendRecv, ScalarTypes);

TYPED_TEST(SendRecv, Contig1D) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 1>(test_exec(), test_utils::Contig{}, "v", 1013);
  test_core_send_recv(v);
}
TYPED_TEST(SendRecv, NonContig1D) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 1>(test_exec(), test_utils::NonContig{}, "v", 1013);
  test_core_send_recv(v);
}
TYPED_TEST(SendRecv, Contig2D) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 2>(test_exec(), test_utils::Contig{}, "v", 137, 17);
  test_core_send_recv(v);
}
TYPED_TEST(SendRecv, NonContig2D) {
  auto v = test_utils::build_view<typename TestFixture::Scalar, 2>(test_exec(), test_utils::NonContig{}, "v", 137, 17);
  test_core_send_recv(v);
}

}  // namespace
