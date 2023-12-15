#include <gtest/gtest.h>

#include <mpi.h>

#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize();

  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  ::testing::TestEventListeners &listeners =
      ::testing::UnitTest::GetInstance()->listeners();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }

  result = RUN_ALL_TESTS();

  Kokkos::finalize();
  MPI_Finalize();

  return result;
}