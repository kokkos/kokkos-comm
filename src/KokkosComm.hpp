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

#include "KokkosComm_include_mpi.hpp"
#include "KokkosComm_collective.hpp"
#include "KokkosComm_version.hpp"
#include "KokkosComm_isend.hpp"
#include "KokkosComm_irecv.hpp"
#include "KokkosComm_recv.hpp"
#include "KokkosComm_send.hpp"
#include "KokkosComm_alltoall.hpp"
#include "KokkosComm_concepts.hpp"
#include "KokkosComm_comm_mode.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdio>
#include <string_view>

namespace KokkosComm {

inline void initialize(int &argc, char ***argv) {
  int flag;
  MPI_Initialized(&flag);
  // Eagerly abort if MPI has already been initialized
  if (0 != flag) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (0 == rank) {
      fprintf(stderr, "error: MPI must not be initialized prior to initializing KokkosComm\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int provided;
  MPI_Init_thread(&argc, argv, MPI_THREAD_MULTIPLE, &provided);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Abort if MPI failed to provide Thread Multiple
  if (MPI_THREAD_MULTIPLE != provided) {
    if (0 == rank) {
      fprintf(stderr, "error: failed to initialized with required thread support\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  // Strip "--help" and "--kokkos-help" from the flags passed to Kokkos if we are not on rank 0 to prevent Kokkos
  // from printing the help message multiple times.
  if (0 != rank) {
    auto *help_it = std::find_if(*argv, *argv + argc,
                                 [](std::string_view const &x) { return x == "--help" || x == "--kokkos-help"; });
    if (help_it != *argv + argc) {
      std::swap(*help_it, *(*argv + argc - 1));
      --argc;
    }
  }
  Kokkos::initialize(argc, *argv);
}

inline void finalize() {
  Kokkos::finalize();
  MPI_Finalize();
}

template <CommMode SendMode = CommMode::Default, KokkosExecutionSpace ExecSpace, KokkosView SendView>
Req isend(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  return Impl::isend<SendMode>(space, sv, dest, tag, comm);
}

template <KokkosView RecvView>
void irecv(RecvView &rv, int src, int tag, MPI_Comm comm, MPI_Request req) {
  return Impl::irecv(rv, src, tag, comm, req);
}

template <CommMode SendMode = CommMode::Default, KokkosExecutionSpace ExecSpace, KokkosView SendView>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  return Impl::send<SendMode>(space, sv, dest, tag, comm);
}

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void recv(const ExecSpace &space, RecvView &rv, int src, int tag, MPI_Comm comm) {
  return Impl::recv(space, rv, src, tag, comm);
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void alltoall(const ExecSpace &space, const SendView &sv, const size_t sendCount, const RecvView &rv,
              const size_t recvCount, MPI_Comm comm) {
  return Impl::alltoall(space, sv, sendCount, rv, recvCount, comm);
}

}  // namespace KokkosComm
