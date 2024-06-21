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

#include <Kokkos_Core.hpp>

#include "KokkosComm_fwd.hpp"
#include "KokkosComm_concepts.hpp"
#include "KokkosComm_plan.hpp"

namespace KokkosComm {

template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
void irecv(Handle<ExecSpace, TRANSPORT> &h, RecvView &rv, int src, int tag) {
  Impl::Irecv<RecvView, ExecSpace, TRANSPORT>(h, rv, src, tag);
}

template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
void isend(Handle<ExecSpace, TRANSPORT> &h, SendView &sv, int dest, int tag) {
  Impl::Isend<SendView, ExecSpace, TRANSPORT>(h, sv, dest, tag);
}

// TODO: can these go in MPI somewhere?
#if defined(KOKKOSCOMM_TRANSPORT_MPI)
template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace>
KokkosComm::Handle<ExecSpace, Mpi> isend(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  return KokkosComm::plan(space, comm,
                          [=](Handle<ExecSpace, Mpi> &handle) { KokkosComm::isend(handle, sv, dest, tag); });
}

template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace>
KokkosComm::Handle<ExecSpace, Mpi> irecv(const ExecSpace &space, const RecvView &rv, int dest, int tag, MPI_Comm comm) {
  return KokkosComm::plan(space, comm,
                          [=](Handle<ExecSpace, Mpi> &handle) { KokkosComm::irecv(handle, rv, dest, tag); });
}

#endif

}  // namespace KokkosComm
