// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>

#include "fwd.hpp"
#include "concepts.hpp"

namespace KokkosComm {

template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
Req<CommSpace> recv(Handle<ExecSpace, CommSpace> &h, RecvView &rv, int src) {
  return Impl::Recv<RecvView, ExecSpace, CommSpace>::execute(h, rv, src);
}

template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
Req<CommSpace> recv(RecvView &rv, int src) {
  return recv<RecvView, ExecSpace, CommSpace>(Handle<ExecSpace, CommSpace>{}, rv, src);
}

template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
Req<CommSpace> send(Handle<ExecSpace, CommSpace> &h, SendView &sv, int dest) {
  return Impl::Send<SendView, ExecSpace, CommSpace>::execute(h, sv, dest);
}

template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
Req<CommSpace> send(SendView &sv, int dest) {
  return send<SendView, ExecSpace, CommSpace>(Handle<ExecSpace, CommSpace>{}, sv, dest);
}

}  // namespace KokkosComm
