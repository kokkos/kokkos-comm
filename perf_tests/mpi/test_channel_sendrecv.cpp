// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include "test_utils.hpp"

#include <KokkosComm/KokkosComm.hpp>

template <typename View>
void channel_send_recv(benchmark::State &, MPI_Comm comm, int rank, const View &v) {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int dest_rank = (rank + 1) % size;         // send to next rank
  const int src_rank  = (rank - 1 + size) % size;  // recv from prev rank
  KokkosComm::Channel<> channel(dest_rank, src_rank, 42, comm);
  channel.sendinit(v);
  channel.recvinit(v);
  channel.start();
  channel.wait();
}

void benchmark_channel_sendrecv(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    state.SkipWithError("benchmark_sendrecv needs at least 2 ranks");
  }

  using Scalar    = double;
  using view_type = Kokkos::View<Scalar *>;
  view_type a("", 1000000);

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, channel_send_recv<view_type>, rank, a);
  }

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * a.size() * 2);
}

BENCHMARK(benchmark_channel_sendrecv)->UseManualTime()->Unit(benchmark::kMillisecond);
