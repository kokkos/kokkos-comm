//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <gtest/gtest.h>

#include "KokkosComm.hpp"

namespace {

template <typename T>
class Allgather : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(Allgather, ScalarTypes);

template <typename Scalar>
void test_allgather_0d() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Kokkos::View<Scalar> sv("sv");
  Kokkos::View<Scalar *> rv("rv", size);

  // fill send buffer
  Kokkos::parallel_for(
      sv.extent(0), KOKKOS_LAMBDA(const int i) { sv() = rank; });

  KokkosComm::allgather(Kokkos::DefaultExecutionSpace(), sv, rv, MPI_COMM_WORLD);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(const int &src, int &lsum) { lsum += rv(src) != src; }, errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(Allgather, 0D) { test_allgather_0d<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_allgather_1d_contig() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int nContrib = 10;

  Kokkos::View<Scalar *> sv("sv", nContrib);
  Kokkos::View<Scalar *> rv("rv", size * nContrib);

  // fill send buffer
  Kokkos::parallel_for(
      sv.extent(0), KOKKOS_LAMBDA(const int i) { sv(i) = rank + i; });

  KokkosComm::allgather(Kokkos::DefaultExecutionSpace(), sv, rv, MPI_COMM_WORLD);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(const int &i, int &lsum) {
        const int src = i / nContrib;
        const int j   = i % nContrib;
        lsum += rv(i) != src + j;
      },
      errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(Allgather, 1D_contig) { test_allgather_1d_contig<typename TestFixture::Scalar>(); }

}  // namespace
