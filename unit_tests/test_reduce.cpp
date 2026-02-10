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
class Reduce : public testing::Test {
 public:
  using Scalar = T;
};
#if defined(KOKKOSCOMM_ENABLE_NCCL)
using ScalarTypes = testing::Types<float, double, int, int64_t>;
#else
using ScalarTypes =
    testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
#endif
TYPED_TEST_SUITE(Reduce, ScalarTypes);

/// Each rank fills its sendbuf[i] with `rank + i`
/// operation is sum, so recvbuf[i] should be sum(0..size) + i * size
template <typename Scalar>
auto reduce_contig_1d() -> void {
#if defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
  GTEST_SKIP() << "Unimplemented test for Open MPI + CUDA/HIP";
#else
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  using ExecSpace = Kokkos::Cuda;
  auto nccl_ctx   = test_utils::nccl::Ctx::init();
  auto space      = ExecSpace(nccl_ctx.stream());
  KokkosComm::Handle<ExecSpace, KokkosComm::Experimental::NcclSpace> h(space, nccl_ctx.comm());
#else
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  auto space      = ExecSpace();
  KokkosComm::Handle<ExecSpace, KokkosComm::MpiSpace> h(space, MPI_COMM_WORLD);
#endif
  int rank = h.rank();
  int size = h.size();
  int root = 0;

  int n_contrib = 10;
  Kokkos::View<Scalar*> sv("sv", n_contrib);
  Kokkos::View<Scalar*> rv("rv", n_contrib);

  // Prepare send buffer
  Kokkos::parallel_for(
      Kokkos::RangePolicy(space, 0, sv.extent(0)), KOKKOS_LAMBDA(const int i) { sv(i) = rank + i; });
  space.fence();
  auto req = KokkosComm::Experimental::reduce(h, sv, rv, root, KokkosComm::Sum{});
  KokkosComm::wait(req);

  if (rank == root) {
    int errs = 0;
    Kokkos::parallel_reduce(
        rv.extent(0),
        KOKKOS_LAMBDA(const int i, int& lsum) { lsum += (rv(i) != ((size * (size - 1)) / 2 + (size * i))); }, errs);
    EXPECT_EQ(errs, 0);
  }
#endif
}

TYPED_TEST(Reduce, Contiguous1D) { reduce_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
