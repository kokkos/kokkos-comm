//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2025) National Technology & Engineering
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
#include <KokkosComm/KokkosComm.hpp>
#include <functional>

using Scalar = double;

// Helper Functions
template <typename Space, typename View>
void lock_unlock_put(
    benchmark::State &, MPI_Comm, const Space &, int rank, const View &v, KokkosComm::Window<View> &window
) {
  if (rank == 0) {
    window.lock(KokkosComm::Window<View>::LockType::Exclusive, 1);
    window.put(v.data(), v.size(), 1, 0);
    window.unlock(1);
  }
}

template <typename Space, typename View>
void lock_unlock_get(
    benchmark::State &, MPI_Comm, const Space &, int rank, const View &v, KokkosComm::Window<View> &window
) {
  if (rank == 1) {
    window.lock(KokkosComm::Window<View>::LockType::Exclusive, 0);
    window.get(v.data(), v.size(), 0, 0);
    window.unlock(0);
  }
}

template <typename Space, typename View>
void lock_unlock_accumulate(
    benchmark::State &, MPI_Comm, const Space &, int rank, const View &v, KokkosComm::Window<View> &window
) {
  if (rank == 0) {
    window.lock(KokkosComm::Window<View>::LockType::Exclusive, 1);
    window.accumulate(v.data(), v.size(), 1, 0, MPI_SUM);
    window.unlock(1);
  }
}

template <typename Space, typename View>
void fence_put(benchmark::State &, MPI_Comm, const Space &, int rank, const View &v, KokkosComm::Window<View> &window) {
  window.fence();
  if (rank == 0) {
    window.put(v.data(), v.size(), 1, 0);
  }
  window.fence();
}

template <typename Space, typename View>
void fence_get(benchmark::State &, MPI_Comm, const Space &, int rank, const View &v, KokkosComm::Window<View> &window) {
  window.fence();
  if (rank == 1) {
    window.get(v.data(), v.size(), 0, 0);
  }
  window.fence();
}

template <typename Space, typename View>
void fence_accumulate(
    benchmark::State &, MPI_Comm, const Space &, int rank, const View &v, KokkosComm::Window<View> &window
) {
  window.fence();
  if (rank == 0) {
    window.accumulate(v.data(), v.size(), 1, 0, MPI_SUM);
  }
  window.fence();
}

template <typename Space, typename View>
void pscw_put(
    benchmark::State &,
    MPI_Comm,
    const Space &,
    int rank,
    const View &v,
    KokkosComm::Window<View> &window,
    MPI_Group &origin_group,
    MPI_Group &target_group
) {
  if (rank == 1) {
    window.post(origin_group);
  }

  if (rank == 0) {
    window.start(target_group);
    window.put(v.data(), v.size(), 1, 0);
    window.complete();
  }

  if (rank == 1) {
    window.wait();
  }
}

template <typename Space, typename View>
void pscw_get(
    benchmark::State &,
    MPI_Comm,
    const Space &,
    int rank,
    const View &v,
    KokkosComm::Window<View> &window,
    MPI_Group &origin_group,
    MPI_Group &target_group
) {
  if (rank == 1) {
    window.post(origin_group);
  }

  if (rank == 0) {
    window.start(target_group);
    window.get(v.data(), v.size(), 1, 0);
    window.complete();
  }

  if (rank == 1) {
    window.wait();
  }
}

template <typename Space, typename View>
void pscw_accumulate(
    benchmark::State &,
    MPI_Comm,
    const Space &,
    int rank,
    const View &v,
    KokkosComm::Window<View> &window,
    MPI_Group &origin_group,
    MPI_Group &target_group
) {
  if (rank == 1) {
    window.post(origin_group);
  }

  if (rank == 0) {
    window.start(target_group);
    window.accumulate(v.data(), v.size(), 1, 0, MPI_SUM);
    window.complete();
  }

  if (rank == 1) {
    window.wait();
  }
}

template <typename Space, typename View>
void sendrecv_comparison(benchmark::State &, MPI_Comm comm, const Space &, int rank, const View &v) {
  if (rank == 0) {
    MPI_Send(v.data(), v.size(), KokkosComm::Impl::mpi_datatype<typename View::value_type>(), 1, 0, comm);
  } else if (rank == 1) {
    MPI_Recv(
        v.data(), v.size(), KokkosComm::Impl::mpi_datatype<typename View::value_type>(), 0, 0, comm, MPI_STATUS_IGNORE
    );
  }
}

// Benchmark Functions
// Lock/Unlock
void benchmark_lock_unlock_put(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    state.SkipWithError("benchmark_lock_unlock_put needs at least 2 ranks");
    return;
  }

  auto space      = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;

  const int n = state.range(0);
  view_type v("data", n);

  Kokkos::parallel_for(
      "init", n, KOKKOS_LAMBDA(int i) { v(i) = static_cast<Scalar>(i); }
  );
  Kokkos::fence();

  KokkosComm::Window<view_type> window(v, MPI_COMM_WORLD);

  while (state.KeepRunning()) {
    do_iteration(
        state, MPI_COMM_WORLD, lock_unlock_put<Kokkos::DefaultExecutionSpace, view_type>, space, rank, v,
        std::ref(window)
    );
  }

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n);
}

void benchmark_lock_unlock_get(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    state.SkipWithError("benchmark_lock_unlock_get needs at least 2 ranks");
    return;
  }

  auto space      = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;

  const int n = state.range(0);
  view_type v("data", n);

  Kokkos::parallel_for(
      "init", n, KOKKOS_LAMBDA(int i) { v(i) = static_cast<Scalar>(i); }
  );
  Kokkos::fence();

  KokkosComm::Window<view_type> window(v, MPI_COMM_WORLD);

  while (state.KeepRunning()) {
    do_iteration(
        state, MPI_COMM_WORLD, lock_unlock_get<Kokkos::DefaultExecutionSpace, view_type>, space, rank, v,
        std::ref(window)
    );
  }

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n);
}

void benchmark_lock_unlock_accumulate(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    state.SkipWithError("benchmark_lock_unlock_accumulate needs at least 2 ranks");
    return;
  }

  auto space      = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;

  const int n = state.range(0);
  view_type v("data", n);

  Kokkos::parallel_for(
      "init", n, KOKKOS_LAMBDA(int i) { v(i) = static_cast<Scalar>(i); }
  );
  Kokkos::fence();

  KokkosComm::Window<view_type> window(v, MPI_COMM_WORLD);

  while (state.KeepRunning()) {
    do_iteration(
        state, MPI_COMM_WORLD, lock_unlock_accumulate<Kokkos::DefaultExecutionSpace, view_type>, space, rank, v,
        std::ref(window)
    );
  }

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n);
}

// Fence
void benchmark_fence_put(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    state.SkipWithError("benchmark_fence_put needs at least 2 ranks");
    return;
  }

  auto space      = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;

  const int n = state.range(0);
  view_type v("data", n);

  Kokkos::parallel_for(
      "init", n, KOKKOS_LAMBDA(int i) { v(i) = static_cast<Scalar>(i); }
  );
  Kokkos::fence();

  KokkosComm::Window<view_type> window(v, MPI_COMM_WORLD);

  while (state.KeepRunning()) {
    do_iteration(
        state, MPI_COMM_WORLD, fence_put<Kokkos::DefaultExecutionSpace, view_type>, space, rank, v, std::ref(window)
    );
  }

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n);
}

void benchmark_fence_get(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    state.SkipWithError("benchmark_fence_get needs at least 2 ranks");
    return;
  }

  auto space      = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;

  const int n = state.range(0);
  view_type v("data", n);

  Kokkos::parallel_for(
      "init", n, KOKKOS_LAMBDA(int i) { v(i) = static_cast<Scalar>(i); }
  );
  Kokkos::fence();

  KokkosComm::Window<view_type> window(v, MPI_COMM_WORLD);

  while (state.KeepRunning()) {
    do_iteration(
        state, MPI_COMM_WORLD, fence_get<Kokkos::DefaultExecutionSpace, view_type>, space, rank, v, std::ref(window)
    );
  }

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n);
}

void benchmark_fence_accumulate(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    state.SkipWithError("benchmark_fence_accumulate needs at least 2 ranks");
    return;
  }

  auto space      = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;

  const int n = state.range(0);
  view_type v("data", n);

  Kokkos::parallel_for(
      "init", n, KOKKOS_LAMBDA(int i) { v(i) = static_cast<Scalar>(i); }
  );
  Kokkos::fence();

  KokkosComm::Window<view_type> window(v, MPI_COMM_WORLD);

  while (state.KeepRunning()) {
    do_iteration(
        state, MPI_COMM_WORLD, fence_accumulate<Kokkos::DefaultExecutionSpace, view_type>, space, rank, v,
        std::ref(window)
    );
  }

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n);
}

// PSCW
void benchmark_pscw_put(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    state.SkipWithError("benchmark_pscw_put needs at least 2 ranks");
    return;
  }

  auto space      = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;

  const int n = state.range(0);
  view_type v("data", n);

  Kokkos::parallel_for(
      "init", n, KOKKOS_LAMBDA(int i) { v(i) = static_cast<Scalar>(i); }
  );
  Kokkos::fence();

  KokkosComm::Window<view_type> window(v, MPI_COMM_WORLD);

  MPI_Group world_group, origin_group, target_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  int origin_rank = 0;
  int target_rank = 1;
  MPI_Group_incl(world_group, 1, &origin_rank, &origin_group);
  MPI_Group_incl(world_group, 1, &target_rank, &target_group);

  while (state.KeepRunning()) {
    do_iteration(
        state, MPI_COMM_WORLD, pscw_put<Kokkos::DefaultExecutionSpace, view_type>, space, rank, v, std::ref(window),
        std::ref(origin_group), std::ref(target_group)
    );
  }

  MPI_Group_free(&origin_group);
  MPI_Group_free(&target_group);
  MPI_Group_free(&world_group);

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n);
}

void benchmark_pscw_get(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    state.SkipWithError("benchmark_pscw_get needs at least 2 ranks");
    return;
  }

  auto space      = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;

  const int n = state.range(0);
  view_type v("data", n);

  Kokkos::parallel_for(
      "init", n, KOKKOS_LAMBDA(int i) { v(i) = static_cast<Scalar>(i); }
  );
  Kokkos::fence();

  KokkosComm::Window<view_type> window(v, MPI_COMM_WORLD);

  MPI_Group world_group, origin_group, target_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  int origin_rank = 0;
  int target_rank = 1;
  MPI_Group_incl(world_group, 1, &origin_rank, &origin_group);
  MPI_Group_incl(world_group, 1, &target_rank, &target_group);

  while (state.KeepRunning()) {
    do_iteration(
        state, MPI_COMM_WORLD, pscw_get<Kokkos::DefaultExecutionSpace, view_type>, space, rank, v, std::ref(window),
        std::ref(origin_group), std::ref(target_group)
    );
  }

  MPI_Group_free(&origin_group);
  MPI_Group_free(&target_group);
  MPI_Group_free(&world_group);

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n);
}

void benchmark_pscw_accumulate(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    state.SkipWithError("benchmark_pscw_accumulate needs at least 2 ranks");
    return;
  }

  auto space      = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;

  const int n = state.range(0);
  view_type v("data", n);

  Kokkos::parallel_for(
      "init", n, KOKKOS_LAMBDA(int i) { v(i) = static_cast<Scalar>(i); }
  );
  Kokkos::fence();

  KokkosComm::Window<view_type> window(v, MPI_COMM_WORLD);

  MPI_Group world_group, origin_group, target_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  int origin_rank = 0;
  int target_rank = 1;
  MPI_Group_incl(world_group, 1, &origin_rank, &origin_group);
  MPI_Group_incl(world_group, 1, &target_rank, &target_group);

  while (state.KeepRunning()) {
    do_iteration(
        state, MPI_COMM_WORLD, pscw_accumulate<Kokkos::DefaultExecutionSpace, view_type>, space, rank, v,
        std::ref(window), std::ref(origin_group), std::ref(target_group)
    );
  }

  MPI_Group_free(&origin_group);
  MPI_Group_free(&target_group);
  MPI_Group_free(&world_group);

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n);
}

// Baseline Comparison
void benchmark_sendrecv_comparison(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    state.SkipWithError("benchmark_sendrecv_comparison needs at least 2 ranks");
    return;
  }

  auto space      = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;

  const int n = state.range(0);
  view_type v("data", n);

  Kokkos::parallel_for(
      "init", n, KOKKOS_LAMBDA(int i) { v(i) = static_cast<Scalar>(i); }
  );
  Kokkos::fence();

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, sendrecv_comparison<Kokkos::DefaultExecutionSpace, view_type>, space, rank, v);
  }

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * n * 2);
}

BENCHMARK(benchmark_lock_unlock_put)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 12)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_lock_unlock_get)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 12)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_lock_unlock_accumulate)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 12)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_fence_put)->RangeMultiplier(8)->Range(1, 1 << 12)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_fence_get)->RangeMultiplier(8)->Range(1, 1 << 12)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_fence_accumulate)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 12)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_pscw_put)->RangeMultiplier(8)->Range(1, 1 << 12)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_pscw_get)->RangeMultiplier(8)->Range(1, 1 << 12)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_pscw_accumulate)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 12)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(benchmark_sendrecv_comparison)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 12)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
