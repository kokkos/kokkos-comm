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

#pragma once

#include <chrono>
#include <benchmark/benchmark.h>
#include "KokkosComm_mpi.hpp"

// F is a function that takes (state, Communicator, args...)
template <typename F, typename... Args>
void do_iteration(benchmark::State &state, KokkosComm::Communicator comm, F &&func, Args... args) {
  using Clock    = std::chrono::steady_clock;
  using Duration = std::chrono::duration<double>;

  auto start = Clock::now();
  func(state, comm, args...);
  Duration elapsed = Clock::now() - start;

  double max_elapsed_seconds;
  double elapsed_seconds = elapsed.count();
  comm.allreduce(Kokkos::View<double>{&elapsed_seconds}, Kokkos::View<double>{&max_elapsed_seconds}, KokkosComm::Max());
  state.SetIterationTime(max_elapsed_seconds);
}