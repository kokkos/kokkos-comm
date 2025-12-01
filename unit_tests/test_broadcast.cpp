// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/utils.hpp"
#endif

namespace {

template <typename T>
class Broadcast : public testing::Test {
 public:
  using Scalar = T;
};
#if defined(KOKKOSCOMM_ENABLE_NCCL)
using ScalarTypes = testing::Types<float, double, int, int64_t>;
#else
using ScalarTypes =
    testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
#endif
TYPED_TEST_SUITE(Broadcast, ScalarTypes);

template <typename Scalar>
auto broadcast_0d() -> void {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  using ExecSpace = Kokkos::Cuda;
  auto nccl_ctx   = test_utils::nccl::Ctx::init();
  KokkosComm::Handle<ExecSpace, KokkosComm::Nccl> h(ExecSpace(), nccl_ctx.comm());
#else
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  KokkosComm::Handle<Kokkos::DefaultExecutionSpace, KokkosComm::Mpi> h{};
#endif
  int rank = h.rank();
  int size = h.size();
  int root = 0;

  Kokkos::View<Scalar> v("v");
  if (rank == root) {
    // Prepare broadcast view
    Kokkos::parallel_for(
        Kokkos::RangePolicy(ExecSpace(), 0, v.extent(0)), KOKKOS_LAMBDA(const int) { v() = size; });
  }
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::Experimental::broadcast(h, v, root);
  KokkosComm::wait(req);

  int errs;
  Kokkos::parallel_reduce(
      v.extent(0), KOKKOS_LAMBDA(const int, int& lsum) { lsum += v() != size; }, errs);
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
auto broadcast_contig_1d() -> void {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  using ExecSpace = Kokkos::Cuda;
  auto nccl_ctx   = test_utils::nccl::Ctx::init();
  KokkosComm::Handle<ExecSpace, KokkosComm::Nccl> h(ExecSpace(), nccl_ctx.comm());
#else
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  KokkosComm::Handle<Kokkos::DefaultExecutionSpace, KokkosComm::Mpi> h{};
#endif
  int rank = h.rank();
  int size = h.size();
  int root = 0;

  Kokkos::View<Scalar*> v("v", 100);
  if (rank == root) {
    // Prepare broadcast view
    Kokkos::parallel_for(
        Kokkos::RangePolicy(ExecSpace(), 0, v.extent(0)), KOKKOS_LAMBDA(const int i) { v(i) = size + i; });
  }
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::Experimental::broadcast(h, v, root);
  KokkosComm::wait(req);

  int errs;
  Kokkos::parallel_reduce(
      v.extent(0), KOKKOS_LAMBDA(const int i, int& lsum) { lsum += (v(i) != size + i); }, errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(Broadcast, 0D) { broadcast_0d<typename TestFixture::Scalar>(); }
TYPED_TEST(Broadcast, Contiguous1D) { broadcast_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
