// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <KokkosComm/KokkosComm.hpp>

namespace {

template <typename T>
class Broadcast : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(Broadcast, ScalarTypes);

template <typename Scalar>
void test_broadcast_0d() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Kokkos::View<Scalar> v("v");

  if (rank == 0) {
    Kokkos::parallel_for(
        v.extent(0), KOKKOS_LAMBDA(int) { v() = size; });
  }

  KokkosComm::mpi::broadcast(Kokkos::DefaultExecutionSpace(), v, 0, MPI_COMM_WORLD);

  int errs;
  Kokkos::parallel_reduce(
      v.extent(0), KOKKOS_LAMBDA(int, int &lsum) { lsum += v() != size; }, errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(Broadcast, 0D) { test_broadcast_0d<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_broadcast_1d_contig() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int nContrib = 10;

  Kokkos::View<Scalar *> v("v", nContrib);

  if (rank == 0) {
    Kokkos::parallel_for(
        v.extent(0), KOKKOS_LAMBDA(int i) { v(i) = size + i; });
  }

  KokkosComm::mpi::broadcast(Kokkos::DefaultExecutionSpace(), v, 0, MPI_COMM_WORLD);

  int errs;
  Kokkos::parallel_reduce(
      v.extent(0), KOKKOS_LAMBDA(int i, int &lsum) { lsum += (v(i) != size + i); }, errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(Broadcast, 1D_contig) { test_broadcast_1d_contig<typename TestFixture::Scalar>(); }

}  // namespace
