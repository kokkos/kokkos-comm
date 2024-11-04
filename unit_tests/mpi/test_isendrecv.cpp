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
class IsendRecv : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes =
    ::testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
TYPED_TEST_SUITE(IsendRecv, ScalarTypes);

template <CommunicationMode IsendMode, typename Scalar>
void isend_comm_mode_1d_contig() {
  if constexpr (std::is_same_v<IsendMode, CommModeReady>) {
    GTEST_SKIP() << "Skipping test for ready-mode send";
  }

  Kokkos::View<Scalar *> a("a", 1000);

  KokkosComm::Handle<> h;
  if (h.size() < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << h.size() << " provided)";
  }

  if (0 == h.rank()) {
    int dst = 1;
    Kokkos::parallel_for(
        a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; });
    KokkosComm::Req req = KokkosComm::mpi::isend(h, a, dst, 0, IsendMode{});
    KokkosComm::wait(req);
  } else if (1 == h.rank()) {
    int src = 0;
    KokkosComm::mpi::recv(h.space(), a, src, 0, h.mpi_comm());
    int errs;
    Kokkos::parallel_reduce(
        a.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += a(i) != Scalar(i); }, errs);
    ASSERT_EQ(errs, 0);
  }
}

template <CommunicationMode IsendMode, typename Scalar>
void isend_comm_mode_1d_noncontig() {
  if constexpr (std::is_same_v<IsendMode, CommModeReady>) {
    GTEST_SKIP() << "Skipping test for ready-mode send";
  }

  // this is C-style layout, i.e. b(0,0) is next to b(0,1)
  Kokkos::View<Scalar **, Kokkos::LayoutRight> b("a", 10, 10);
  auto a = Kokkos::subview(b, Kokkos::ALL, 2);  // take column 2 (non-contiguous)

  KokkosComm::Handle<> h;
  if (h.size() < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << h.size() << " provided)";
  }

  if (0 == h.rank()) {
    int dst = 1;
    Kokkos::parallel_for(
        a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; });
    KokkosComm::Req req = KokkosComm::mpi::isend(h, a, dst, 0, IsendMode{});
    KokkosComm::wait(req);
  } else if (1 == h.rank()) {
    int src = 0;
    KokkosComm::mpi::recv(h.space(), a, src, 0, h.mpi_comm());
    int errs;
    Kokkos::parallel_reduce(
        a.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += a(i) != Scalar(i); }, errs);
    ASSERT_EQ(errs, 0);
  }
}

TYPED_TEST(IsendRecv, 1D_contig_standard) {
  isend_comm_mode_1d_contig<CommModeStandard, typename TestFixture::Scalar>();
}

TYPED_TEST(IsendRecv, 1D_contig_ready) { isend_comm_mode_1d_contig<CommModeReady, typename TestFixture::Scalar>(); }

TYPED_TEST(IsendRecv, 1D_contig_synchronous) {
  isend_comm_mode_1d_contig<CommModeSynchronous, typename TestFixture::Scalar>();
}

TYPED_TEST(IsendRecv, 1D_noncontig_standard) {
  isend_comm_mode_1d_noncontig<CommModeStandard, typename TestFixture::Scalar>();
}

TYPED_TEST(IsendRecv, 1D_noncontig_ready) {
  isend_comm_mode_1d_noncontig<CommModeReady, typename TestFixture::Scalar>();
}

TYPED_TEST(IsendRecv, 1D_noncontig_synchronous) {
  isend_comm_mode_1d_noncontig<CommModeSynchronous, typename TestFixture::Scalar>();
}

}  // namespace
