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

#include <iostream>

void noop(benchmark::State, MPI_Comm) {}

template <CommunicationMode Mode, typename Space, typename View>
void send_recv(benchmark::State &, MPI_Comm comm, const Mode &mode, const Space &space, int nx, int ny, int rx, int ry,
               int rs, const View &v) {
  // 2D index of nbrs in minus and plus direction (periodic)
  const int xm1 = (rx + rs - 1) % rs;
  const int ym1 = (ry + rs - 1) % rs;
  const int xp1 = (rx + 1) % rs;
  const int yp1 = (ry + 1) % rs;

  // convert 2D rank into 1D rank
  auto get_rank = [=](const int x, const int y) -> int { return y * rs + x; };

  auto make_pair = [](int a, int b) -> Kokkos::pair<int, int> { return Kokkos::pair{a, b}; };

  // send/recv subviews
  auto xp1_s = Kokkos::subview(v, v.extent(0) - 2, make_pair(1, ny + 1), Kokkos::ALL);
  auto xp1_r = Kokkos::subview(v, v.extent(0) - 1, make_pair(1, ny + 1), Kokkos::ALL);
  auto xm1_s = Kokkos::subview(v, 1, make_pair(1, ny + 1), Kokkos::ALL);
  auto xm1_r = Kokkos::subview(v, 0, make_pair(1, ny + 1), Kokkos::ALL);
  auto yp1_s = Kokkos::subview(v, make_pair(1, nx + 1), v.extent(1) - 2, Kokkos::ALL);
  auto yp1_r = Kokkos::subview(v, make_pair(1, nx + 1), v.extent(1) - 1, Kokkos::ALL);
  auto ym1_s = Kokkos::subview(v, make_pair(1, nx + 1), 1, Kokkos::ALL);
  auto ym1_r = Kokkos::subview(v, make_pair(1, nx + 1), 0, Kokkos::ALL);

  std::vector<KokkosComm::Req> reqs;
  // std::cerr << get_rank(rx, ry) << " -> " << get_rank(xp1, ry) << "\n";
  reqs.push_back(KokkosComm::isend(mode, space, xp1_s, get_rank(xp1, ry), 0, comm));
  reqs.push_back(KokkosComm::isend(mode, space, xm1_s, get_rank(xm1, ry), 1, comm));
  reqs.push_back(KokkosComm::isend(mode, space, yp1_s, get_rank(rx, yp1), 2, comm));
  reqs.push_back(KokkosComm::isend(mode, space, ym1_s, get_rank(rx, ym1), 3, comm));

  KokkosComm::recv(space, xm1_r, get_rank(xm1, ry), 0, comm);
  KokkosComm::recv(space, xp1_r, get_rank(xp1, ry), 1, comm);
  KokkosComm::recv(space, ym1_r, get_rank(rx, ym1), 2, comm);
  KokkosComm::recv(space, yp1_r, get_rank(rx, yp1), 3, comm);

  // wait for comm
  for (KokkosComm::Req &req : reqs) {
    req.wait();
  }
}

void benchmark_2dhalo(benchmark::State &state) {
  using Scalar    = double;
  using grid_type = Kokkos::View<Scalar ***, Kokkos::LayoutRight>;

  // problem size per rank
  int nx     = 512;
  int ny     = 512;
  int nprops = 3;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int rs = std::sqrt(size);
  const int rx = rank % rs;
  const int ry = rank / rs;

  if (rank < rs * rs) {
    auto mode  = KokkosComm::DefaultCommMode();
    auto space = Kokkos::DefaultExecutionSpace();
    // grid of elements, each with 3 properties, and a radius-1 halo
    grid_type grid("", nx + 2, ny + 2, nprops);
    while (state.KeepRunning()) {
      do_iteration(state, MPI_COMM_WORLD,
                   send_recv<KokkosComm::DefaultCommMode, Kokkos::DefaultExecutionSpace, grid_type>, mode, space, nx,
                   ny, rx, ry, rs, grid);
    }
  } else {
    while (state.KeepRunning()) {
      do_iteration(state, MPI_COMM_WORLD, noop);  // do nothing...
    }
  }

  state.counters["active_ranks"] = rs * rs;
  state.counters["nx"]           = nx;
  // clang-format off
  state.SetBytesProcessed(
      sizeof(Scalar) 
    * rs * rs // active ranks
    * state.iterations()
    * nprops
    * (
        2 * nx // send x nbrs
      + 2 * nx // recv x nbs
      + 2 * ny // send y nbrs
      + 2 * ny // recv y nbs
    )
  );
  // clang-format on
}

BENCHMARK(benchmark_2dhalo)->UseManualTime()->Unit(benchmark::kMillisecond);
