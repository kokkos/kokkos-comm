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

#include "test_utils.hpp"

#include "KokkosComm.hpp"

template <typename Space, typename View>
void send_recv(benchmark::State &, const KokkosComm::Communicator<Space> &comm, const Space &space, int rank,
               const View &v) {
  if (0 == rank) {
    KokkosComm::send(space, v, 1, 0, comm);
    KokkosComm::recv(space, v, 1, 0, comm);
  } else if (1 == rank) {
    KokkosComm::recv(space, v, 0, 0, comm);
    KokkosComm::send(space, v, 0, 0, comm);
  }
}

void benchmark_sendrecv(benchmark::State &state) {
  auto ctx         = KokkosComm::initialize<Kokkos::DefaultExecutionSpace>();
  const auto &comm = ctx.comm();

  int rank = comm.rank();
  int size = comm.size();
  if (size < 2) {
    state.SkipWithError("benchmark_sendrecv needs at least 2 ranks");
  }

  using Scalar = double;

  auto space      = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;
  view_type a("", 1000000);

  while (state.KeepRunning()) {
    do_iteration(state, comm, send_recv<Kokkos::DefaultExecutionSpace, view_type>, space, rank, a);
  }

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * a.size() * 2);
}

BENCHMARK(benchmark_sendrecv)->UseManualTime()->Unit(benchmark::kMillisecond);
