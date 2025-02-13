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

#include "KokkosComm/KokkosComm.hpp"

#include "../view_builder.hpp"

namespace {

template <typename T>
class Reduce : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(Reduce, ScalarTypes);

namespace {
/*!
Each rank fills its sendbuf[i] with `rank + i`

operation is sum, so recvbuf[i] should be sum(0..size) + i * size
*/
template <typename Scalar, typename SendContig, typename RecvContig>
void test_reduce_1d() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto sendv = ViewBuilder<Scalar, 1>::view(SendContig{}, "sendv", 65536);
  auto recvv = ViewBuilder<Scalar, 1>::view(RecvContig{}, "recvv", 65536);

  // fill send buffer
  Kokkos::parallel_for(
      sendv.extent(0), KOKKOS_LAMBDA(const int i) { sendv(i) = rank + i; });

  KokkosComm::mpi::reduce(Kokkos::DefaultExecutionSpace{}, sendv, recvv, MPI_SUM, 0, MPI_COMM_WORLD);

  if (0 == rank) {
    int errs;
    Kokkos::parallel_reduce(
        recvv.extent(0),
        KOKKOS_LAMBDA(const int &i, int &lsum) {
          Scalar acc = 0;
          for (int r = 0; r < size; ++r) {
            acc += r + i;  // every rank contributes rank + i
          }
          lsum += recvv(i) != acc;
        },
        errs);
    ASSERT_EQ(errs, 0);
  }
}

}  // namespace

TYPED_TEST(Reduce, 1D_contig_contig) { test_reduce_1d<typename TestFixture::Scalar, contig, contig>(); }
TYPED_TEST(Reduce, 1D_contig_noncontig) { test_reduce_1d<typename TestFixture::Scalar, contig, noncontig>(); }
TYPED_TEST(Reduce, 1D_noncontig_contig) { test_reduce_1d<typename TestFixture::Scalar, noncontig, contig>(); }
TYPED_TEST(Reduce, 1D_noncontig_noncontig) { test_reduce_1d<typename TestFixture::Scalar, noncontig, noncontig>(); }

}  // namespace
