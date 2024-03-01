// Copyright 2023 Carl Pearson
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if KOKKOSCOMM_EXPERIMENTAL_MDSPAN
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

#include <gtest/gtest.h>

#include "KokkosComm.hpp"

template <typename T> class IsendRecv : public testing::Test {
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
  auto a = Kokkos::subview(b, 2, Kokkos::ALL); // take column 2 (non-contiguous)

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

TYPED_TEST(IsendRecv, 1D_mdspan_noncontig) {
  using ScalarType = typename TestFixture::Scalar;

  // this is C-style layout, i.e. b(0,0) is next to b(0,1)
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

#endif // KOKKOSCOMM_ENABLE_MDSPAN