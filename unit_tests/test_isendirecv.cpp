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
#include "KokkosComm_irecv.hpp"

#include "view_builder.hpp"

namespace {

template <typename T>
class IsendIrecv : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes =
    ::testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
TYPED_TEST_SUITE(IsendIrecv, ScalarTypes);

template <KokkosComm::KokkosView View1D>
void test_1d(const View1D &a) {
  static_assert(View1D::rank == 1, "");
  using Scalar = typename View1D::non_const_value_type;

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
    KokkosComm::Req req =
        KokkosComm::isend(KokkosComm::CommMode::Default{}, Kokkos::DefaultExecutionSpace(), a, dst, 0, MPI_COMM_WORLD);
    req.wait();
  } else if (1 == rank) {
    int src = 0;
    MPI_Request req;
    KokkosComm::irecv(a, src, 0, MPI_COMM_WORLD, req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    int errs;
    Kokkos::parallel_reduce(
        a.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += a(i) != Scalar(i); }, errs);
    ASSERT_EQ(errs, 0);
  }
}

template <KokkosComm::KokkosView View2D>
void test_2d(const View2D &a) {
  static_assert(View2D::rank == 2, "");
  using Scalar = typename View2D::non_const_value_type;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }

  using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
  Policy policy({0, 0}, {a.extent(0), a.extent(1)});

  if (0 == rank) {
    int dst = 1;
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(int i, int j) { a(i, j) = i * a.extent(0) + j; });
    KokkosComm::Req req =
        KokkosComm::isend(KokkosComm::CommMode::Default{}, Kokkos::DefaultExecutionSpace(), a, dst, 0, MPI_COMM_WORLD);
    req.wait();
  } else if (1 == rank) {
    int src             = 0;
    KokkosComm::Req req = KokkosComm::irecv(a, src, 0, MPI_COMM_WORLD);
    req.wait();
    int errs;
    Kokkos::parallel_reduce(
        policy, KOKKOS_LAMBDA(int i, int j, int &lsum) { lsum += a(i, j) != Scalar(i * a.extent(0) + j); }, errs);
    ASSERT_EQ(errs, 0);
  }
}

TYPED_TEST(IsendIrecv, 1D_contig) {
  auto a = ViewBuilder<typename TestFixture::Scalar, 1>::view(contig{}, "a", 1013);
  test_1d(a);
}

TYPED_TEST(IsendIrecv, 2D_contig) {
  auto a = ViewBuilder<typename TestFixture::Scalar, 2>::view(contig{}, "a", 137, 17);
  test_2d(a);
}

}  // namespace
