// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core_fwd.hpp>

#include "fwd.hpp"
#if defined(KOKKOSCOMM_ENABLE_MPI)
#include "mpi/isend.hpp"
#include "mpi/irecv.hpp"
#endif
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/send.hpp"
#include "nccl/recv.hpp"
#endif

namespace KokkosComm {

/// Send w/ explicit handle
template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto send(Handle<ExecSpace, CommSpace> &h, SendView &sv, int peer) -> Req<CommSpace> {
  return Impl::Send<SendView, ExecSpace, CommSpace>::execute(h, sv, peer);
}

/// Send w/ implicit handle
template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto send(SendView &sv, int peer) -> Req<CommSpace> {
  return send<SendView, ExecSpace, CommSpace>(Handle<ExecSpace, CommSpace>{}, sv, peer);
}

/// Receive w/ explicit handle
template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto recv(Handle<ExecSpace, CommSpace> &h, RecvView &rv, int peer) -> Req<CommSpace> {
  return Impl::Recv<RecvView, ExecSpace, CommSpace>::execute(h, rv, peer);
}

/// Receive w/ implicit handle
template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto recv(RecvView &rv, int peer) -> Req<CommSpace> {
  return recv<RecvView, ExecSpace, CommSpace>(Handle<ExecSpace, CommSpace>{}, rv, peer);
}

}  // namespace KokkosComm
