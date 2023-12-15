#include "test_utils.hpp"

#include <thread>

void i_am_sleepy(MPI_Comm comm, int macsleepface) {
  // Pretend to work ...
  std::this_thread::sleep_for(std::chrono::milliseconds(macsleepface));
  // ... as a team
  MPI_Barrier(MPI_COMM_WORLD);
}



void mpi_benchmark(benchmark::State &state) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  while(state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, i_am_sleepy, rank % 5);
  }
}

BENCHMARK(mpi_benchmark)->UseManualTime()->Unit(benchmark::kMillisecond);;