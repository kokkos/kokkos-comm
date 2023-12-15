#include <gtest/gtest.h>

#include "KokkosComm.hpp"

TEST(HelloTest, BasicAssertions) {

  Kokkos::View<float *> a("a", 1000);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (0 == rank) {
    int dst = 1;
    KokkosComm::send(Kokkos::DefaultExecutionSpace(), a, dst, 0,
                     MPI_COMM_WORLD);
  } else if (1 == rank) {
    int src = 0;
    KokkosComm::recv(Kokkos::DefaultExecutionSpace(), a, src, 0,
                     MPI_COMM_WORLD);
  }
}
