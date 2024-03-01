#include <mpi.h>

#include <iostream>

int main(int argc, char *argv[]) {
  // Initialize MPI before any call to gtest_mpi
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::cerr << "Hello from rank " << rank << "/" << size << "\n";
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}