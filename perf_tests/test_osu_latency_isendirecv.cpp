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

// Adapted from the OSU Benchmarks
// Copyright (c) 2002-2024 the Network-Based Computing Laboratory
// (NBCL), The Ohio State University.

#include "test_utils.hpp"
#include "KokkosComm.hpp"

template <typename Mode, typename Space, typename View>
void osu_latency_Kokkos_Comm_isendirecv(benchmark::State &, MPI_Comm comm, const Mode &mode, const Space &space,
                                        int rank, const View &v) {
  if (rank == 0) {
    KokkosComm::Req sendreq = KokkosComm::isend(mode, space, v, 1, 1, comm);
    sendreq.wait();
  } else if (rank == 1) {
    KokkosComm::Req recvreq = KokkosComm::irecv(v, 0, 1, comm);
    recvreq.wait();
  }
}

template <typename View>
void osu_latency_MPI_isendirecv(benchmark::State &, MPI_Comm comm, int rank, const View &v) {
  MPI_Request sendreq, recvreq;
  if (rank == 0) {
    MPI_Irecv(v.data(), v.size(), KokkosComm::Impl::mpi_type<typename View::value_type>(), 1, 0, comm, &recvreq);
    MPI_Wait(&recvreq, MPI_STATUS_IGNORE);
  } else if (rank == 1) {
    MPI_Isend(v.data(), v.size(), KokkosComm::Impl::mpi_type<typename View::value_type>(), 0, 0, comm, &sendreq);
    MPI_Wait(&sendreq, MPI_STATUS_IGNORE);
  }
}

void benchmark_osu_latency_KokkosComm_isendirecv(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    state.SkipWithError("benchmark_osu_latency_KokkosComm needs exactly 2 ranks");
  }

  auto mode       = KokkosComm::DefaultCommMode();
  auto space      = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<char *>;
  view_type a("A", state.range(0));

  while (state.KeepRunning()) {
    do_iteration(
        state, MPI_COMM_WORLD,
        osu_latency_Kokkos_Comm_isendirecv<KokkosComm::DefaultCommMode, Kokkos::DefaultExecutionSpace, view_type>, mode,
        space, rank, a);
  }
  state.counters["bytes"] = a.size() * 2;
}

void benchmark_osu_latency_MPI_isendirecv(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    state.SkipWithError("benchmark_osu_latency_MPI needs exactly 2 ranks");
  }

  using view_type = Kokkos::View<char *>;
  view_type a("A", state.range(0));

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, osu_latency_MPI_isendirecv<view_type>, rank, a);
  }
  state.counters["bytes"] = a.size() * 2;
}

BENCHMARK(benchmark_osu_latency_KokkosComm_isendirecv)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 28);
BENCHMARK(benchmark_osu_latency_MPI_isendirecv)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 28);
