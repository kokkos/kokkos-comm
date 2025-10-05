// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <KokkosComm/KokkosComm.hpp>

namespace {

template <typename T>
class Allreduce : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(Allreduce, ScalarTypes);

template <typename Scalar>
void test_allreduce_0d() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Kokkos::View<Scalar> sv("sv");
  Kokkos::View<Scalar> rv("rv");

  // fill send buffer
  Kokkos::parallel_for(
      sv.extent(0), KOKKOS_LAMBDA(int) { sv() = rank; });

  KokkosComm::mpi::allreduce(Kokkos::DefaultExecutionSpace(), sv, rv, MPI_SUM, MPI_COMM_WORLD);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(int, int &lsum) { lsum += (rv() != size * (size - 1) / 2); }, errs);
  EXPECT_EQ(errs, 0);

  KokkosComm::mpi::allreduce(Kokkos::DefaultExecutionSpace(), sv, MPI_SUM, MPI_COMM_WORLD);

  Kokkos::parallel_reduce(
      sv.extent(0), KOKKOS_LAMBDA(int, int &lsum) { lsum += (sv() != size * (size - 1) / 2); }, errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(Allreduce, 0D) { test_allreduce_0d<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_allreduce_1d_contig() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int nContrib = 10;

  Kokkos::View<Scalar *> sv("sv", nContrib);
  Kokkos::View<Scalar *> rv("rv", nContrib);

  // fill send buffer
  Kokkos::parallel_for(
      sv.extent(0), KOKKOS_LAMBDA(int const i) { sv(i) = rank + i; });

  KokkosComm::mpi::allreduce(Kokkos::DefaultExecutionSpace(), sv, rv, MPI_SUM, MPI_COMM_WORLD);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(int i, int &lsum) { lsum += (rv(i) != size * (size - 1) / 2 + size * i); }, errs);
  EXPECT_EQ(errs, 0);

  KokkosComm::mpi::allreduce(Kokkos::DefaultExecutionSpace(), sv, MPI_SUM, MPI_COMM_WORLD);

  Kokkos::parallel_reduce(
      sv.extent(0), KOKKOS_LAMBDA(int i, int &lsum) { lsum += (sv(i) != size * (size - 1) / 2 + size * i); }, errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(Allreduce, 1D_contig) { test_allreduce_1d_contig<typename TestFixture::Scalar>(); }

}  // namespace
