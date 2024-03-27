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

#if KOKKOSCOMM_ENABLE_MDSPAN
#if KOKKOSCOMM_MDSPAN_IN_EXPERIMENTAL
#include <experimental/mdspan>
#define MDSPAN_PREFIX() experimental::
#else
#include <mdspan>
#define MDSPAN_PREFIX()
#endif

using std::MDSPAN_PREFIX() dextents;
using std::MDSPAN_PREFIX() extents;
using std::MDSPAN_PREFIX() layout_stride;
using std::MDSPAN_PREFIX() mdspan;
#endif  // KOKKOSCOMM_ENABLE_MDSPAN

#include <gtest/gtest.h>

#include <KokkosComm.hpp>

template <typename T>
class IsendRecv : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes =
    ::testing::Types<float, double, Kokkos::complex<float>,
                     Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
TYPED_TEST_SUITE(IsendRecv, ScalarTypes);

TYPED_TEST(IsendRecv, 1D_contig) {
  Kokkos::View<typename TestFixture::Scalar *> a("a", 1000);

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
    KokkosComm::Req req = KokkosComm::isend(Kokkos::DefaultExecutionSpace(), a,
                                            dst, 0, MPI_COMM_WORLD);
    req.wait();
  } else if (1 == rank) {
    int src = 0;
    KokkosComm::recv(Kokkos::DefaultExecutionSpace(), a, src, 0,
                     MPI_COMM_WORLD);
    int errs;
    Kokkos::parallel_reduce(
        a.extent(0),
        KOKKOS_LAMBDA(const int &i, int &lsum) {
          lsum += a(i) != typename TestFixture::Scalar(i);
        },
        errs);
    ASSERT_EQ(errs, 0);
  }
}

TYPED_TEST(IsendRecv, 1D_noncontig) {
  // this is C-style layout, i.e. b(0,0) is next to b(0,1)
  Kokkos::View<typename TestFixture::Scalar **, Kokkos::LayoutRight> b("a", 10,
                                                                       10);
  auto a =
      Kokkos::subview(b, Kokkos::ALL, 2);  // take column 2 (non-contiguous)

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (0 == rank) {
    int dst = 1;
    Kokkos::parallel_for(
        a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; });
    KokkosComm::Req req = KokkosComm::isend(Kokkos::DefaultExecutionSpace(), a,
                                            dst, 0, MPI_COMM_WORLD);
    req.wait();
  } else if (1 == rank) {
    int src = 0;
    KokkosComm::recv(Kokkos::DefaultExecutionSpace(), a, src, 0,
                     MPI_COMM_WORLD);
    int errs;
    Kokkos::parallel_reduce(
        a.extent(0),
        KOKKOS_LAMBDA(const int &i, int &lsum) {
          lsum += a(i) != typename TestFixture::Scalar(i);
        },
        errs);
    ASSERT_EQ(errs, 0);
  }
}

#if KOKKOSCOMM_ENABLE_MDSPAN

TYPED_TEST(IsendRecv, 1D_mdspan_contig) {
  using ScalarType = typename TestFixture::Scalar;

  std::vector<ScalarType> v(100);
  auto a = mdspan(&v[2], 13);  // 13 scalars starting at index 2

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (0 == rank) {
    int dst = 1;
    for (size_t i = 0; i < a.extent(0); ++i) {
      a[i] = i;
    }
    KokkosComm::Req req = KokkosComm::isend(Kokkos::DefaultExecutionSpace(), a,
                                            dst, 0, MPI_COMM_WORLD);
    req.wait();
  } else if (1 == rank) {
    int src = 0;
    KokkosComm::recv(Kokkos::DefaultExecutionSpace(), a, src, 0,
                     MPI_COMM_WORLD);
    int errs = 0;
    for (size_t i = 0; i < a.extent(0); ++i) {
      errs += (a[i] != ScalarType(i));
    }
    ASSERT_EQ(errs, 0);
  }
}

TYPED_TEST(IsendRecv, 1D_mdspan_noncontig) {
  using ScalarType = typename TestFixture::Scalar;

  std::vector<ScalarType> v(100);

  using ExtentsType = dextents<std::size_t, 1>;
  ExtentsType shape{10};
  std::array<std::size_t, 1> strides{10};

  mdspan<ScalarType, ExtentsType, layout_stride> a(
      &v[2], layout_stride::mapping{shape, strides});

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (0 == rank) {
    int dst = 1;
    for (size_t i = 0; i < a.extent(0); ++i) {
      a[i] = i;
    }
    KokkosComm::Req req = KokkosComm::isend(Kokkos::DefaultExecutionSpace(), a,
                                            dst, 0, MPI_COMM_WORLD);
    req.wait();
  } else if (1 == rank) {
    int src = 0;
    KokkosComm::recv(Kokkos::DefaultExecutionSpace(), a, src, 0,
                     MPI_COMM_WORLD);
    int errs = 0;
    for (size_t i = 0; i < a.extent(0); ++i) {
      errs += (a[i] != ScalarType(i));
    }
    ASSERT_EQ(errs, 0);
  }
}

#endif  // KOKKOSCOMM_ENABLE_MDSPAN