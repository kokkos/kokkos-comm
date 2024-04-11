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

#include "KokkosComm_collective.hpp"
#include "KokkosComm_version.hpp"
#include "KokkosComm_isend.hpp"
#include "KokkosComm_recv.hpp"
#include "KokkosComm_send.hpp"
#include "KokkosComm_concepts.hpp"
#include "KokkosComm_mode.hpp"

#include <Kokkos_Core.hpp>

namespace KokkosComm {

template <KokkosExecutionSpace ExecSpace,KokkosView SendView,
          Mode CommMode = Mode::Default>
          KokkosView SendView>
Req isend(const ExecSpace &space, const SendView &sv, int dest, int tag,
          MPI_Comm comm) {
  if constexpr (CommMode == Mode::Default) {
    return Impl::isend(space, sv, dest, tag, comm);
  } else if constexpr (CommMode == Mode::Ready) {
    return Impl::irsend(space, sv, dest, tag, comm);
  } else if constexpr (CommMode == Mode::Synchronous) {
    return Impl::issend(space, sv, dest, tag, comm);
  }
}

template <KokkosExecutionSpace ExecSpace,KokkosView SendView,
          Mode CommMode = Mode::Default>
          KokkosView SendView>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag,
          MPI_Comm comm) {
  if constexpr (CommMode == Mode::Default) {
    return Impl::send(space, sv, dest, tag, comm);
  } else if constexpr (CommMode == Mode::Ready) {
    return Impl::rsend(space, sv, dest, tag, comm);
  } else if constexpr (CommMode == Mode::Synchronous) {
    return Impl::ssend(space, sv, dest, tag, comm);
  }
}

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void recv(const ExecSpace &space, RecvView &sv, int src, int tag,
          MPI_Comm comm) {
  return Impl::recv(space, sv, src, tag, comm);
}

}  // namespace KokkosComm
