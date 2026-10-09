// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstdint>

#include <gtest/gtest.h>
#include <nccl.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/KokkosComm.hpp>

#include "utils.hpp"

namespace {

using ExecSpace = Kokkos::Cuda;
using CommSpace = KokkosComm::Experimental::NcclSpace;

template <typename T>
class PointToPoint : public testing::Test {
 public:
  using Scalar = T;
};
using ScalarTypes = ::testing::Types<int, int64_t, float, double>;

TYPED_TEST_SUITE(PointToPoint, ScalarTypes);

template <typename Scalar>
auto p2p_contig_1d() -> void {
  auto& nccl_ctx  = test_utils::NcclCtx::get();
  const auto exec = Kokkos::Cuda(nccl_ctx.stream());
  const auto comm = nccl_ctx.comm();
  const int size  = nccl_ctx.size();
  const int rank  = nccl_ctx.rank();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  int src = 0;
  int dst = 1;

  Kokkos::View<Scalar*> v("v", 10'000);
  if (rank == src) {
    // Prepare send view
    Kokkos::parallel_for(
        Kokkos::RangePolicy(exec, 0, v.extent(0)), KOKKOS_LAMBDA(const int i) { v(i) = i; }
    );

    // Using the same execution space for both operations lets us not need an explicit `fence`
    KokkosComm::Experimental::nccl::send(exec, v, dst, comm).wait();
  } else if (rank == dst) {
    KokkosComm::Experimental::nccl::recv(exec, v, src, comm).wait();

    int errs;
    Kokkos::parallel_reduce(
        v.extent(0), KOKKOS_LAMBDA(const int i, int& lsum) { lsum += v(i) != Scalar(i); }, errs
    );
    ASSERT_EQ(errs, 0);
  }
}

template <typename Scalar>
auto p2p_noncontig_1d() -> void {
  auto& nccl_ctx  = test_utils::NcclCtx::get();
  const auto exec = Kokkos::Cuda(nccl_ctx.stream());
  const auto comm = nccl_ctx.comm();
  const int size  = nccl_ctx.size();
  const int rank  = nccl_ctx.rank();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  int src = 0;
  int dst = 1;

  Kokkos::View<Scalar**, Kokkos::LayoutRight> v("v", 100, 100);
  auto sv = Kokkos::subview(v, Kokkos::ALL, 2);  // take column 2 (non-contiguous)
  if (rank == src) {
    // Prepare send view
    Kokkos::parallel_for(
        Kokkos::RangePolicy(exec, 0, sv.extent(0)), KOKKOS_LAMBDA(const int i) { sv(i) = i; }
    );

    // Using the same execution space for both operations lets us not need an explicit `fence`
    KokkosComm::Experimental::nccl::send(exec, sv, dst, comm).wait();
  } else if (rank == dst) {
    KokkosComm::Experimental::nccl::recv(exec, sv, src, comm).wait();

    int errs;
    Kokkos::parallel_reduce(
        sv.extent(0), KOKKOS_LAMBDA(const int i, int& lsum) { lsum += sv(i) != Scalar(i); }, errs
    );
    ASSERT_EQ(errs, 0);
  }
}

TYPED_TEST(PointToPoint, Contiguous1D) { p2p_contig_1d<typename TestFixture::Scalar>(); }
TYPED_TEST(PointToPoint, NonContiguous1D) { p2p_noncontig_1d<typename TestFixture::Scalar>(); }

auto p2p_cuda_error() -> void {
  auto& nccl_ctx  = test_utils::NcclCtx::get();
  const auto comm = nccl_ctx.comm();
  const int size  = nccl_ctx.size();
  const int rank  = nccl_ctx.rank();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  const auto exec = Kokkos::Cuda(stream);
  Kokkos::View<double*> v("v", 10'000);

  // Both ranks: spin ~1s on the GPU, then fault. The NCCL op queued behind this never
  // runs on either side, so neither rank hangs waiting for its peer.
  Kokkos::parallel_for(
      Kokkos::RangePolicy(exec, 0, 1),
      KOKKOS_LAMBDA(int) {
        KOKKOS_IF_ON_DEVICE((const long long start = clock64();
                             while (clock64() - start < 2'000'000'000LL) {} Kokkos::abort("injected CUDA error");))
      }
  );

  auto request = (rank == 0) ? KokkosComm::Experimental::nccl::send(exec, v, 1, comm)
                             : KokkosComm::Experimental::nccl::recv(exec, v, 0, comm);
  EXPECT_FALSE(request.has_error());  // enqueued well before the kernel faults

  request.wait();
  EXPECT_TRUE(request.has_error());
  EXPECT_EQ(request.error_code(), KokkosComm::ErrorCode::CudaError);

  // The context is now dead: skip Kokkos/NCCL teardown, which would abort.
  std::fflush(nullptr);
  std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
}

TEST(PointToPointError, CudaErrorDuringWait) { p2p_cuda_error(); }

}  // namespace
