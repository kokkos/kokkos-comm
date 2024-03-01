#pragma once

#include <chrono>

#include <benchmark/benchmark.h>
#include <mpi.h>

// F is a function that takes (state, MPI_Comm, args...)
template <typename F, typename... Args>
void do_iteration(benchmark::State &state, MPI_Comm comm, F &&func,
                  Args... args) {

  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::duration<double>;

  auto start = Clock::now();
  func(state, comm, args...);
  Duration elapsed = Clock::now() - start;

  double max_elapsed_second;
  double elapsed_seconds = elapsed.count();
  MPI_Allreduce(&elapsed_seconds, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX,
                comm);
  state.SetIterationTime(max_elapsed_second);
}