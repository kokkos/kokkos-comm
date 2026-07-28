// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <chrono>

#include <benchmark/benchmark.h>
#include <mpi.h>

template <typename F, typename... Args>
auto do_iteration(benchmark::State& state, F&& func, Args&&... args) -> void {
  using Clock    = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double>;

  auto start = Clock::now();
  func(state, std::forward<Args>(args)...);
  Duration elapsed = Clock::now() - start;

  double max_elapsed_second;
  double elapsed_seconds = elapsed.count();
  MPI_Allreduce(&elapsed_seconds, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  state.SetIterationTime(max_elapsed_second);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    state.counters["rank0_ping_pong_latency"] = elapsed_seconds;
  }
}
