#include <gtest/gtest.h>

#include "KokkosComm.hpp"


template <typename T>
class Reduce : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(Reduce, ScalarTypes);

TYPED_TEST(Reduce, 1D_contig) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Kokkos::View<typename TestFixture::Scalar *> sendv("sendv", 10);
  Kokkos::View<typename TestFixture::Scalar *> recvv;
  if (0 == rank) {
    Kokkos::resize(recvv, sendv.extent(0) * size);
  }

  // fill send buffer
  Kokkos::parallel_for(sendv.extent(0), KOKKOS_LAMBDA(const int i){ sendv(i) = rank + i; });

  KokkosComm::reduce(Kokkos::DefaultExecutionSpace(), sendv, recvv, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if (0 == rank) {
    int errs;
    Kokkos::parallel_reduce(recvv.extent(0), KOKKOS_LAMBDA (const int& i, int& lsum) {
      const int idx = i % sendv.extent(0);
      typename TestFixture::Scalar acc = 0;
      for (int j = 0; j < size; ++j) {
        acc += j + idx;
      }
      lsum += recvv(i) != acc;
    }, errs);
    ASSERT_EQ(errs, 0);
  }
}

