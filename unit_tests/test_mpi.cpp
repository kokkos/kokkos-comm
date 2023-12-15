#include <mpi.h>

#include <iostream>

int main(int argc, char *argv[]) {
  // Initialize MPI before any call to gtest_mpi
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cerr << "Hello from rank " << rank << "\n";
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}