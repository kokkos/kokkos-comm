// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>
#include <mpi.h>
#ifdef KOKKOSCOMM_ENABLE_NCCL
#include <nccl.h>
#endif

#include "utils.hpp"
#ifdef KOKKOSCOMM_ENABLE_NCCL
#include "../unit_tests/nccl/utils.hpp"  // For test_utils::NcclCtx
#endif

namespace KC  = KokkosComm;
namespace KCE = KC::Experimental;

using DES = Kokkos::DefaultExecutionSpace;
#ifdef KOKKOSCOMM_ENABLE_NCCL
// If NCCL is enabled, then `KOKKOS_ENABLE_CUDA` is defined.
using CuES = Kokkos::Cuda;
#endif

using View_t = Kokkos::View<char*>;
// --- Benchmark kernels ---

template <class Comm, KC::KokkosView View>
void lat_KC(benchmark::State&, Comm& comm, const View& sv, const View& rv) {
  if (comm.rank() == 0) {
    auto s_req = KC::send(comm, sv, 1);
    auto r_req = KC::recv(comm, rv, 1);
    s_req.wait();
    r_req.wait();
  } else {
    auto r_req = KC::recv(comm, rv, 0);
    // Actually wait for the ping reception before sending the pong
    r_req.wait();
    auto s_req = KC::send(comm, sv, 0);
    s_req.wait();
  }
}

template <KC::KokkosView View>
void lat_KC_mpi_nb(benchmark::State&, KC::Communicator<KC::MpiSpace, DES>& comm, const View& sv, const View& rv) {
  if (comm.rank() == 0) {
    auto s_req = KC::mpi::isend(comm.exec(), sv, 1, 0, comm.comm());
    auto r_req = KC::mpi::irecv(comm.exec(), rv, 1, 1, comm.comm());
    s_req.wait();
    r_req.wait();
  } else {
    auto r_req = KC::mpi::irecv(comm.exec(), rv, 0, 0, comm.comm());
    // Actually wait for the ping reception before sending the pong
    r_req.wait();
    auto s_req = KC::mpi::isend(comm.exec(), sv, 0, 1, comm.comm());
    s_req.wait();
  }
}

template <KC::KokkosView View>
void lat_KC_mpi(benchmark::State&, KC::Communicator<KC::MpiSpace, DES>& comm, const View& sv, const View& rv) {
  if (comm.rank() == 0) {
    KC::mpi::send(comm.exec(), sv, 1, 0, comm.comm());
    KC::mpi::recv(comm.exec(), rv, 1, 1, comm.comm());
  } else {
    KC::mpi::recv(comm.exec(), rv, 0, 0, comm.comm());
    KC::mpi::send(comm.exec(), sv, 0, 1, comm.comm());
  }
}

template <KC::KokkosView View>
void lat_MPI_nb(benchmark::State&, MPI_Comm comm, const View& sv, const View& rv, int rank) {
  MPI_Request s_req, r_req;
  if (rank == 0) {
    MPI_Isend(sv.data(), sv.size(), MPI_CHAR, 1, 0, comm, &s_req);
    MPI_Irecv(rv.data(), rv.size(), MPI_CHAR, 1, 1, comm, &r_req);
    MPI_Wait(&s_req, MPI_STATUS_IGNORE);
    MPI_Wait(&r_req, MPI_STATUS_IGNORE);
  } else {
    MPI_Irecv(rv.data(), rv.size(), MPI_CHAR, 0, 0, comm, &r_req);
    // Actually wait for the ping reception before sending the pong
    MPI_Wait(&r_req, MPI_STATUS_IGNORE);
    MPI_Isend(sv.data(), sv.size(), MPI_CHAR, 0, 1, comm, &s_req);
    MPI_Wait(&s_req, MPI_STATUS_IGNORE);
  }
}

template <KC::KokkosView View>
void lat_MPI(benchmark::State&, MPI_Comm comm, const View& sv, const View& rv, int rank) {
  if (rank == 0) {
    MPI_Send(sv.data(), sv.size(), MPI_CHAR, 1, 0, comm);
    MPI_Recv(rv.data(), rv.size(), MPI_CHAR, 1, 1, comm, MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(rv.data(), rv.size(), MPI_CHAR, 0, 0, comm, MPI_STATUS_IGNORE);
    MPI_Send(sv.data(), sv.size(), MPI_CHAR, 0, 1, comm);
  }
}

#ifdef KOKKOSCOMM_ENABLE_NCCL
template <KC::KokkosView View>
void lat_KC_nccl(benchmark::State&, KC::Communicator<KCE::NcclSpace, DES>& comm, const View& sv, const View& rv) {
  if (comm.rank() == 0) {
    auto s_req = KCE::nccl::send(comm.exec(), sv, 1, comm.comm());
    auto r_req = KCE::nccl::recv(comm.exec(), rv, 1, comm.comm());
    s_req.wait();
    r_req.wait();
  } else {
    auto r_req = KCE::nccl::recv(comm.exec(), rv, 0, comm.comm());
    // Actually wait for the ping reception before sending the pong
    r_req.wait();
    auto s_req = KCE::nccl::send(comm.exec(), sv, 0, comm.comm());
    s_req.wait();
  }
}

template <KC::KokkosView View>
void lat_KC_nccl_stream_ordered(
    benchmark::State&, KC::Communicator<KCE::NcclSpace, DES>& comm, const View& sv, const View& rv
) {
  if (comm.rank() == 0) {
    auto s_req = KCE::nccl::send(comm.exec(), sv, 1, comm.comm());
    auto r_req = KCE::nccl::recv(comm.exec(), rv, 1, comm.comm());
    s_req.wait();
    r_req.wait();
  } else {
    auto r_req = KCE::nccl::recv(comm.exec(), rv, 0, comm.comm());
    // Stream-ordered semantics mean we don't need to explicitly wait for the recv to complete: the send cannot
    // start until the reception is not complete. r_req.wait();
    auto s_req = KCE::nccl::send(comm.exec(), sv, 0, comm.comm());
    s_req.wait();
  }
}

template <KC::KokkosView View>
void lat_NCCL(benchmark::State&, const View& sv, const View& rv, int rank, ncclComm_t comm, cudaStream_t stream) {
  if (rank == 0) {
    ncclSend(sv.data(), sv.size(), ncclChar, 1, comm, stream);
    ncclRecv(rv.data(), rv.size(), ncclChar, 1, comm, stream);
    cudaStreamSynchronize(stream);
  } else if (rank == 1) {
    ncclRecv(rv.data(), rv.size(), ncclChar, 0, comm, stream);
    // Actually wait for the ping reception before sending the pong
    cudaStreamSynchronize(stream);
    ncclSend(sv.data(), sv.size(), ncclChar, 0, comm, stream);
    cudaStreamSynchronize(stream);
  }
}

template <KC::KokkosView View>
void lat_NCCL_stream_ordered(
    benchmark::State&, const View& sv, const View& rv, int rank, ncclComm_t comm, cudaStream_t stream
) {
  if (rank == 0) {
    ncclSend(sv.data(), sv.size(), ncclChar, 1, comm, stream);
    ncclRecv(rv.data(), rv.size(), ncclChar, 1, comm, stream);
    cudaStreamSynchronize(stream);
  } else if (rank == 1) {
    ncclRecv(rv.data(), rv.size(), ncclChar, 0, comm, stream);
    // Stream-ordered semantics mean we don't need to explicitly wait for the recv to complete: the send cannot
    // start until the reception is not complete. r_req.wait();
    ncclSend(sv.data(), sv.size(), ncclChar, 0, comm, stream);
    cudaStreamSynchronize(stream);
  }
}
#endif

// --- Benchmark drivers ---

auto bench_KC_MpiSpace(benchmark::State& state) -> void {
  auto comm = KC::Communicator<KC::MpiSpace, DES>::from_raw(MPI_COMM_WORLD, DES{});
  if (comm.size() != 2) {
    state.SkipWithError("KokkosComm::MpiSpace latency benchmark needs exactly 2 ranks");
  }
  View_t sv("sv", state.range(0));
  View_t rv("rv", state.range(0));
  for (auto _ : state) {
    do_iteration(state, lat_KC<decltype(comm), View_t>, comm, sv, rv);
  }
  state.counters["bytes"] = sv.size() + rv.size();
}

auto bench_KC_mpi(benchmark::State& state) -> void {
  auto comm = KC::Communicator<KC::MpiSpace, DES>::from_raw(MPI_COMM_WORLD, DES{});
  if (comm.size() != 2) {
    state.SkipWithError("KokkosComm::mpi latency benchmark needs exactly 2 ranks");
  }
  View_t sv("sv", state.range(0));
  View_t rv("rv", state.range(0));
  for (auto _ : state) {
    do_iteration(state, lat_KC_mpi<View_t>, comm, sv, rv);
  }
  state.counters["bytes"] = sv.size() + rv.size();
}

auto bench_MPI(benchmark::State& state) -> void {
  auto comm      = MPI_COMM_WORLD;
  const int size = [comm]() {
    int _s;
    MPI_Comm_size(comm, &_s);
    return _s;
  }();
  const int rank = [comm]() {
    int _r;
    MPI_Comm_rank(comm, &_r);
    return _r;
  }();
  if (size != 2) {
    state.SkipWithError("MPI latency benchmark needs exactly 2 ranks");
  }
  View_t sv("sv", state.range(0));
  View_t rv("rv", state.range(0));
  for (auto _ : state) {
    do_iteration(state, lat_MPI<View_t>, comm, sv, rv, rank);
  }
  state.counters["bytes"] = sv.size() + rv.size();
}

auto bench_MPI_nb(benchmark::State& state) -> void {
  auto comm      = MPI_COMM_WORLD;
  const int size = [comm]() {
    int _s;
    MPI_Comm_size(comm, &_s);
    return _s;
  }();
  const int rank = [comm]() {
    int _r;
    MPI_Comm_rank(comm, &_r);
    return _r;
  }();
  if (size != 2) {
    state.SkipWithError("MPI nonblocking latency benchmark needs exactly 2 ranks");
  }
  View_t sv("sv", state.range(0));
  View_t rv("rv", state.range(0));
  for (auto _ : state) {
    do_iteration(state, lat_MPI_nb<View_t>, comm, sv, rv, rank);
  }
  state.counters["bytes"] = sv.size() + rv.size();
}

#ifdef KOKKOSCOMM_ENABLE_NCCL
auto bench_KC_NcclSpace(benchmark::State& state) -> void {
  auto& nccl_ctx = test_utils::NcclCtx::get();
  auto comm      = KC::Communicator<KCE::NcclSpace, CuES>::from_raw(nccl_ctx.comm(), CuES{});
  if (comm.size() != 2) {
    state.SkipWithError("KokkosComm::Experimental::NcclSpace latency benchmark needs exactly 2 ranks");
  }
  View_t sv("sv", state.range(0));
  View_t rv("rv", state.range(0));
  for (auto _ : state) {
    do_iteration(state, lat_KC<decltype(comm), View_t>, comm, sv, rv);
  }
  state.counters["bytes"] = sv.size() + rv.size();
}

auto bench_KC_nccl(benchmark::State& state) -> void {
  auto& nccl_ctx = test_utils::NcclCtx::get();
  auto comm      = KC::Communicator<KCE::NcclSpace, CuES>::from_raw(nccl_ctx.comm(), CuES{});
  if (comm.size() != 2) {
    state.SkipWithError("KokkosComm::Experimental::nccl latency benchmark needs exactly 2 ranks");
  }
  View_t sv("sv", state.range(0));
  View_t rv("rv", state.range(0));
  for (auto _ : state) {
    do_iteration(state, lat_KC_nccl<View_t>, comm, sv, rv);
  }
  state.counters["bytes"] = sv.size() + rv.size();
}

auto bench_KC_nccl_stream_ordered(benchmark::State& state) -> void {
  auto& nccl_ctx = test_utils::NcclCtx::get();
  auto comm      = KC::Communicator<KCE::NcclSpace, CuES>::from_raw(nccl_ctx.comm(), CuES{});
  if (comm.size() != 2) {
    state.SkipWithError("KokkosComm::Experimental::nccl latency benchmark needs exactly 2 ranks");
  }
  View_t sv("sv", state.range(0));
  View_t rv("rv", state.range(0));
  for (auto _ : state) {
    do_iteration(state, lat_KC_nccl_stream_ordered<View_t>, comm, sv, rv);
  }
  state.counters["bytes"] = sv.size() + rv.size();
}

auto bench_NCCL(benchmark::State& state) -> void {
  auto& nccl_ctx      = test_utils::NcclCtx::get();
  auto comm           = nccl_ctx.comm();
  const int size      = nccl_ctx.size();
  const int rank      = nccl_ctx.rank();
  cudaStream_t stream = nccl_ctx.stream();
  if (size != 2) {
    state.SkipWithError("NCCL latency benchmark needs exactly 2 ranks");
  }
  View_t sv("sv", state.range(0));
  View_t rv("rv", state.range(0));
  for (auto _ : state) {
    do_iteration(state, lat_NCCL<View_t>, sv, rv, rank, comm, stream);
  }
  state.counters["bytes"] = sv.size() + rv.size();
}

auto bench_NCCL_stream_ordered(benchmark::State& state) -> void {
  auto& nccl_ctx      = test_utils::NcclCtx::get();
  auto comm           = nccl_ctx.comm();
  const int size      = nccl_ctx.size();
  const int rank      = nccl_ctx.rank();
  cudaStream_t stream = nccl_ctx.stream();
  if (size != 2) {
    state.SkipWithError("NCCL latency benchmark needs exactly 2 ranks");
  }
  View_t sv("sv", state.range(0));
  View_t rv("rv", state.range(0));
  for (auto _ : state) {
    do_iteration(state, lat_NCCL_stream_ordered<View_t>, sv, rv, rank, comm, stream);
  }
  state.counters["bytes"] = sv.size() + rv.size();
}
#endif

// --- Benchmark decls ---

BENCHMARK(bench_KC_MpiSpace)->UseManualTime()->Unit(benchmark::kMicrosecond)->RangeMultiplier(8)->Range(1, 1 << 28);
BENCHMARK(bench_KC_mpi)->UseManualTime()->Unit(benchmark::kMicrosecond)->RangeMultiplier(8)->Range(1, 1 << 28);
BENCHMARK(bench_MPI)->UseManualTime()->Unit(benchmark::kMicrosecond)->RangeMultiplier(8)->Range(1, 1 << 28);
BENCHMARK(bench_MPI_nb)->UseManualTime()->Unit(benchmark::kMicrosecond)->RangeMultiplier(8)->Range(1, 1 << 28);
#ifdef KOKKOSCOMM_ENABLE_NCCL
BENCHMARK(bench_KC_NcclSpace)->UseManualTime()->Unit(benchmark::kMicrosecond)->RangeMultiplier(8)->Range(1, 1 << 28);
BENCHMARK(bench_KC_nccl)->UseManualTime()->Unit(benchmark::kMicrosecond)->RangeMultiplier(8)->Range(1, 1 << 28);
// BENCHMARK(bench_KC_nccl_stream_ordered)->UseManualTime()->Unit(benchmark::kMicrosecond)->RangeMultiplier(8)->Range(1,
// 1 << 28);
BENCHMARK(bench_NCCL)->UseManualTime()->Unit(benchmark::kMicrosecond)->RangeMultiplier(8)->Range(1, 1 << 28);
// BENCHMARK(bench_NCCL_stream_ordered)->UseManualTime()->Unit(benchmark::kMicrosecond)->RangeMultiplier(8)->Range(1, 1
// << 28);
#endif
