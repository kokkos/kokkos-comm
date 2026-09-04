// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include "test_utils.hpp"

#include <cstdint>
#include <vector>

#include <KokkosComm/KokkosComm.hpp>

void run_channel_p2p_cycle(benchmark::State&, MPI_Comm, KokkosComm::Channel<>& channel) {
  channel.start();
  channel.wait();
}

void benchmark_channel_p2p(benchmark::State& state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    state.SkipWithError("benchmark_channel_p2p needs at least 2 ranks");
    return;
  }

  const auto buffer_size     = state.range(0);
  const auto operation_pairs = state.range(1);
  const int dst_rank         = (rank + 1) % size;
  const int src_rank         = (rank - 1 + size) % size;

  using view_type = Kokkos::View<char*>;
  std::vector<view_type> send_buffers;
  std::vector<view_type> recv_buffers;
  send_buffers.reserve(operation_pairs);
  recv_buffers.reserve(operation_pairs);

  KokkosComm::Channel<> channel(dst_rank, src_rank, 42, MPI_COMM_WORLD);
  for (std::int64_t operation = 0; operation < operation_pairs; ++operation) {
    send_buffers.emplace_back("channel_send", buffer_size);
    recv_buffers.emplace_back("channel_recv", buffer_size);
    channel.sendinit(send_buffers.back());
    channel.recvinit(recv_buffers.back());
  }

  for (auto _ : state) {
    do_iteration(state, MPI_COMM_WORLD, run_channel_p2p_cycle, channel);
  }

  state.counters["registered_operations"] = 2 * operation_pairs;
  state.SetBytesProcessed(state.iterations() * buffer_size * operation_pairs * 2);
}

BENCHMARK(benchmark_channel_p2p)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({benchmark::CreateRange(1, 1 << 23, 8), benchmark::CreateRange(1, 16, 2)})
    ->ArgNames({"bytes", "operation_pairs"});
