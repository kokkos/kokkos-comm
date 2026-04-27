// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <type_traits>

#include <gtest/gtest.h>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

namespace {

template <typename Scalar>
void test_inclusive_scan_0d() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Kokkos::View<Scalar> sv("sv");
  Kokkos::View<Scalar> rv("rv");

  // fill send buffer
  Kokkos::parallel_for(
      sv.extent(0), KOKKOS_LAMBDA(int) { sv() = rank; }
  );

  KokkosComm::mpi::inclusive_scan(Kokkos::DefaultExecutionSpace(), sv, rv, MPI_SUM, MPI_COMM_WORLD);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(int, int &lsum) { lsum += (rv() != rank * (rank + 1) / 2); }, errs
  );
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
void test_exclusive_scan_0d() {
  // FIXME_EXTERNAL #204, #226
#if defined(KOKKOSCOMM_IMPL_MPI_IS_MPICH) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
  if constexpr (std::is_same_v<Scalar, double> or std::is_same_v<Scalar, Kokkos::complex<float>> or std::is_same_v<Scalar, Kokkos::complex<double>>) {
    GTEST_SKIP() << "Unimplemented test for MPICH + CUDA/HIP";
  }
#else
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Kokkos::View<Scalar> sv("sv");
  Kokkos::View<Scalar> rv("rv");

  // fill send buffer
  Kokkos::parallel_for(
      sv.extent(0), KOKKOS_LAMBDA(int) { sv() = rank; }
  );

  KokkosComm::mpi::exclusive_scan(Kokkos::DefaultExecutionSpace(), sv, rv, MPI_SUM, MPI_COMM_WORLD);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(int, int &lsum) { lsum += (rv() != rank * (rank - 1) / 2); }, errs
  );
  EXPECT_EQ(errs, 0);
#endif
}

template <typename Scalar>
void test_inclusive_scan_1d_contig() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int nContrib = 10;

  Kokkos::View<Scalar *> sv("sv", nContrib);
  Kokkos::View<Scalar *> rv("rv", nContrib);

  // fill send buffer
  Kokkos::parallel_for(
      sv.extent(0), KOKKOS_LAMBDA(int const i) { sv(i) = rank + i; }
  );

  KokkosComm::mpi::inclusive_scan(Kokkos::DefaultExecutionSpace(), sv, rv, MPI_SUM, MPI_COMM_WORLD);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(int i, int &lsum) { lsum += (rv(i) != (rank + i) * (rank + i + 1) / 2 - i * (i - 1) / 2); }, errs
  );
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
void test_exclusive_scan_1d_contig() {
  // FIXME_EXTERNAL #204, #226
#if defined(KOKKOSCOMM_IMPL_MPI_IS_MPICH) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
  if constexpr (std::is_same_v<Scalar, double> or std::is_same_v<Scalar, Kokkos::complex<float>> or std::is_same_v<Scalar, Kokkos::complex<double>>) {
    GTEST_SKIP() << "Unimplemented test for MPICH + CUDA/HIP";
  }
#else
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int nContrib = 10;

  Kokkos::View<Scalar *> sv("sv", nContrib);
  Kokkos::View<Scalar *> rv("rv", nContrib);

  // fill send buffer
  Kokkos::parallel_for(
      sv.extent(0), KOKKOS_LAMBDA(int const i) { sv(i) = rank + i; }
  );

  KokkosComm::mpi::exclusive_scan(Kokkos::DefaultExecutionSpace(), sv, rv, MPI_SUM, MPI_COMM_WORLD);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(int i, int &lsum) { lsum += (rv(i) != (rank + i) * (rank + i - 1) / 2 - i * (i - 1) / 2); }, errs
  );
  EXPECT_EQ(errs, 0);
#endif
}

template <typename T>
class Scan : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(Scan, ScalarTypes);

TYPED_TEST(Scan, Inclusive0D) { test_inclusive_scan_0d<typename TestFixture::Scalar>(); }
TYPED_TEST(Scan, Exclusive0D) { test_exclusive_scan_0d<typename TestFixture::Scalar>(); }

TYPED_TEST(Scan, InclusiveContig1D) { test_inclusive_scan_1d_contig<typename TestFixture::Scalar>(); }
TYPED_TEST(Scan, ExclusiveContig1D) { test_exclusive_scan_1d_contig<typename TestFixture::Scalar>(); }

}  // namespace
