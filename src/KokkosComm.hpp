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

#include <Kokkos_Core.hpp>

namespace KokkosComm {

// Scoped enumeration to specify the communication mode of a sending operation.
// See section 3.4 of the MPI standard for a complete specification.
enum class CommMode : uint8_t {
  // Standard mode: MPI implementation decides whether outgoing messages will
  // be buffered. Send operations can be started whether or not a matching
  // receive has been started. They may complete before a matching receive is
  // started. Standard mode is non-local: successful completion of the send
  // operation may depend on the occurrence of a matching receive.
  Standard,
  // Ready mode: Send operations may be started only if the matching receive is
  // already started.
  Ready,
  // Synchronous mode: Send operations complete successfully only if a matching
  // receive is started, and the receive operation has started to receive the
  // message sent.
  Synchronous,
};

template <CommMode SendMode = CommMode::Standard,
          KokkosExecutionSpace ExecSpace, KokkosView SendView>
Req isend(const ExecSpace &space, const SendView &sv, int dest, int tag,
          MPI_Comm comm) {
  if constexpr (SendMode == CommMode::Standard) {
    return Impl::isend(space, sv, dest, tag, comm);
  } else if constexpr (SendMode == CommMode::Ready) {
    return Impl::irsend(space, sv, dest, tag, comm);
  } else if constexpr (SendMode == CommMode::Synchronous) {
    return Impl::issend(space, sv, dest, tag, comm);
  }
}

template <CommMode SendMode = CommMode::Standard,
          KokkosExecutionSpace ExecSpace, KokkosView SendView>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag,
          MPI_Comm comm) {
  if constexpr (SendMode == CommMode::Standard) {
    return Impl::send(space, sv, dest, tag, comm);
  } else if constexpr (SendMode == CommMode::Ready) {
    return Impl::rsend(space, sv, dest, tag, comm);
  } else if constexpr (SendMode == CommMode::Synchronous) {
    return Impl::ssend(space, sv, dest, tag, comm);
  }
}

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void recv(const ExecSpace &space, RecvView &sv, int src, int tag,
          MPI_Comm comm) {
  return Impl::recv(space, sv, src, tag, comm);
}

}  // namespace KokkosComm
