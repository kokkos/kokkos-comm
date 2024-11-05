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

#include <KokkosComm/KokkosComm.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using namespace KokkosComm::mpi;

template <typename T>
class MpiSendRecv : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(MpiSendRecv, ScalarTypes);

template <CommunicationMode SendMode, typename Scalar>
void send_comm_mode_1d_contig() {
  if constexpr (std::is_same_v<SendMode, CommModeReady>) {
    GTEST_SKIP() << "Skipping test for ready-mode send";
  }

  Kokkos::View<Scalar *> a("a", 1000);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }

  if (0 == rank) {
    int dst = 1;
    Kokkos::parallel_for(
        a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; });
    KokkosComm::mpi::send(Kokkos::DefaultExecutionSpace(), a, dst, 0, MPI_COMM_WORLD, SendMode{});
  } else if (1 == rank) {
    int src = 0;
    KokkosComm::mpi::recv(Kokkos::DefaultExecutionSpace(), a, src, 0, MPI_COMM_WORLD);
    int errs;
    Kokkos::parallel_reduce(
        a.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += a(i) != i; }, errs);
    ASSERT_EQ(errs, 0);
  }
}

template <CommunicationMode SendMode, typename Scalar>
void send_comm_mode_1d_noncontig() {
  if constexpr (std::is_same_v<SendMode, CommModeReady>) {
    GTEST_SKIP() << "Skipping test for ready-mode send";
  }

  // this is C-style layout, i.e. b(0,0) is next to b(0,1)
  Kokkos::View<Scalar **, Kokkos::LayoutRight> b("b", 10, 10);
  auto a = Kokkos::subview(b, Kokkos::ALL, 2);  // take column 2 (non-contiguous)

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (0 == rank) {
    int dst = 1;
    Kokkos::parallel_for(
        a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; });
    KokkosComm::mpi::send(Kokkos::DefaultExecutionSpace(), a, dst, 0, MPI_COMM_WORLD, SendMode{});
  } else if (1 == rank) {
    int src = 0;
    KokkosComm::mpi::recv(Kokkos::DefaultExecutionSpace(), a, src, 0, MPI_COMM_WORLD);
    int errs;
    Kokkos::parallel_reduce(
        a.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += a(i) != i; }, errs);
    ASSERT_EQ(errs, 0);
  }
}

TYPED_TEST(MpiSendRecv, 1D_contig_standard) {
  send_comm_mode_1d_contig<CommModeStandard, typename TestFixture::Scalar>();
}

TYPED_TEST(MpiSendRecv, 1D_contig_ready) { send_comm_mode_1d_contig<CommModeReady, typename TestFixture::Scalar>(); }

TYPED_TEST(MpiSendRecv, 1D_contig_synchronous) {
  send_comm_mode_1d_contig<CommModeSynchronous, typename TestFixture::Scalar>();
}

TYPED_TEST(MpiSendRecv, 1D_noncontig_standard) {
  send_comm_mode_1d_noncontig<CommModeStandard, typename TestFixture::Scalar>();
}

TYPED_TEST(MpiSendRecv, 1D_noncontig_ready) {
  send_comm_mode_1d_noncontig<CommModeReady, typename TestFixture::Scalar>();
}

TYPED_TEST(MpiSendRecv, 1D_noncontig_synchronous) {
  send_comm_mode_1d_noncontig<CommModeSynchronous, typename TestFixture::Scalar>();
}

}  // namespace
