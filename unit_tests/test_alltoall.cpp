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
class Alltoall : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(Alltoall, ScalarTypes);

TYPED_TEST(Alltoall, 1D_contig) {
  using TestScalar = typename TestFixture::Scalar;

  auto comm = KokkosComm::CommWorld();
  int rank  = comm.rank();
  int size  = comm.size();

  const int nContrib = 10;

  Kokkos::View<TestScalar *> sv("sv", size * nContrib);
  Kokkos::View<TestScalar *> rv("rv", size * nContrib);

  // fill send buffer
  Kokkos::parallel_for(
      sv.extent(0), KOKKOS_LAMBDA(const int i) { sv(i) = rank + i; });

  KokkosComm::Impl::alltoall(Kokkos::DefaultExecutionSpace(), sv, nContrib, rv, nContrib, comm);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(const int &i, int &lsum) {
        const int src = i / nContrib;                      // who sent this data
        const int j   = rank * nContrib + (i % nContrib);  // what index i was at the source
        lsum += rv(i) != src + j;
      },
      errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(Alltoall, 1D_inplace_contig) {
  using TestScalar = typename TestFixture::Scalar;

  auto comm = KokkosComm::CommWorld();
  int rank  = comm.rank();
  int size  = comm.size();

  const int nContrib = 10;

  Kokkos::View<TestScalar *> rv("rv", size * nContrib);

  // fill send buffer
  Kokkos::parallel_for(
      rv.extent(0), KOKKOS_LAMBDA(const int i) { rv(i) = rank + i; });

  KokkosComm::Impl::alltoall(Kokkos::DefaultExecutionSpace(), rv, nContrib, comm);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(const int &i, int &lsum) {
        const int src = i / nContrib;                      // who sent this data
        const int j   = rank * nContrib + (i % nContrib);  // what index i was at the source
        lsum += rv(i) != src + j;
      },
      errs);
  EXPECT_EQ(errs, 0);
}

}  // namespace
