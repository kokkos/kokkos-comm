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
#include <Kokkos_Core_fwd.hpp>

#include "KokkosComm.hpp"

namespace {

template <typename T>
class SendRecv : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(SendRecv, ScalarTypes);

template <KokkosComm::CommMode SendMode, typename Scalar>
void send_comm_mode_1d_contig() {
  if (SendMode == KokkosComm::CommMode::Ready) {
    GTEST_SKIP() << "Skipping test for ready-mode send";
  }

  Kokkos::View<Scalar *> a("a", 1000);

  auto ctx         = KokkosComm::initialize<Kokkos::DefaultExecutionSpace>();
  const auto &comm = ctx.comm();

  int rank = comm.rank();
  int size = comm.size();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }

  if (0 == rank) {
    int dst = 1;
    Kokkos::parallel_for(
        a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; });
    KokkosComm::send<SendMode>(Kokkos::DefaultExecutionSpace(), a, dst, 0, comm);
  } else if (1 == rank) {
    int src = 0;
    KokkosComm::recv(Kokkos::DefaultExecutionSpace(), a, src, 0, comm);
    int errs;
    Kokkos::parallel_reduce(
        a.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += a(i) != i; }, errs);
    ASSERT_EQ(errs, 0);
  }
}

template <KokkosComm::CommMode SendMode, typename Scalar>
void send_comm_mode_1d_noncontig() {
  if (SendMode == KokkosComm::CommMode::Ready) {
    GTEST_SKIP() << "Skipping test for ready-mode send";
  }

  // this is C-style layout, i.e. b(0,0) is next to b(0,1)
  Kokkos::View<Scalar **, Kokkos::LayoutRight> b("b", 10, 10);
  auto a = Kokkos::subview(b, Kokkos::ALL, 2);  // take column 2 (non-contiguous)

  auto ctx         = KokkosComm::initialize<Kokkos::DefaultExecutionSpace>();
  const auto &comm = ctx.comm();

  int rank = comm.rank();
  if (0 == rank) {
    int dst = 1;
    Kokkos::parallel_for(
        a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; });
    KokkosComm::send<SendMode>(Kokkos::DefaultExecutionSpace(), a, dst, 0, comm);
  } else if (1 == rank) {
    int src = 0;
    KokkosComm::recv(Kokkos::DefaultExecutionSpace(), a, src, 0, comm);
    int errs;
    Kokkos::parallel_reduce(
        a.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += a(i) != i; }, errs);
    ASSERT_EQ(errs, 0);
  }
}

TYPED_TEST(SendRecv, 1D_contig_standard) {
  send_comm_mode_1d_contig<KokkosComm::CommMode::Standard, typename TestFixture::Scalar>();
}

TYPED_TEST(SendRecv, 1D_contig_ready) {
  send_comm_mode_1d_contig<KokkosComm::CommMode::Ready, typename TestFixture::Scalar>();
}

TYPED_TEST(SendRecv, 1D_contig_synchronous) {
  send_comm_mode_1d_contig<KokkosComm::CommMode::Synchronous, typename TestFixture::Scalar>();
}

TYPED_TEST(SendRecv, 1D_noncontig_standard) {
  send_comm_mode_1d_noncontig<KokkosComm::CommMode::Standard, typename TestFixture::Scalar>();
}

TYPED_TEST(SendRecv, 1D_noncontig_ready) {
  send_comm_mode_1d_noncontig<KokkosComm::CommMode::Ready, typename TestFixture::Scalar>();
}

TYPED_TEST(SendRecv, 1D_noncontig_synchronous) {
  send_comm_mode_1d_noncontig<KokkosComm::CommMode::Synchronous, typename TestFixture::Scalar>();
}

}  // namespace
