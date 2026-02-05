// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <KokkosComm/KokkosComm.hpp>

namespace {

template <typename T>
class NonBlockingReduce : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(NonBlockingReduce, ScalarTypes);

template <typename Scalar>
void test_ireduce_1d_contig() {
#if defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
  GTEST_SKIP() << "Unimplemented test for Open MPI + CUDA/HIP";
#else
  Kokkos::DefaultExecutionSpace space{};

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int root = 0;

  int n_contrib = 10;
  Kokkos::View<Scalar*> sv("sv", n_contrib);
  Kokkos::View<Scalar*> rv("rv", n_contrib);

  // Prepare send buffer
  Kokkos::parallel_for(
      Kokkos::RangePolicy(space, 0, sv.extent(0)), KOKKOS_LAMBDA(const int i) { sv(i) = rank + i; });
  space.fence();

  auto req = KokkosComm::mpi::ireduce(space, sv, rv, MPI_SUM, root, MPI_COMM_WORLD);
  KokkosComm::wait(req);

  if (root == rank) {
    int errs;
    Kokkos::parallel_reduce(
        rv.extent(0),
        KOKKOS_LAMBDA(const int i, int& lsum) {
          Scalar acc = 0;
          for (int r = 0; r < size; ++r) {
            acc += r + i;
          }
          lsum += rv(i) != acc;
        },
        errs);
    ASSERT_EQ(errs, 0);
  }
#endif
}

TYPED_TEST(NonBlockingReduce, 1D_contig) { test_ireduce_1d_contig<typename TestFixture::Scalar>(); }

}  // namespace
