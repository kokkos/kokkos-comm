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

#include <KokkosComm/nccl/nccl.hpp>
#include <KokkosComm/nccl/send.hpp>

#include <KokkosComm/fwd.hpp>
#include <KokkosComm/concepts.hpp>

#include <Kokkos_Core_fwd.hpp>

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

namespace Experimental {

template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::Cuda, CommunicationSpace CommSpace = Nccl>
auto send(const Handle<ExecSpace, CommSpace>& h, const SendView& sv, int dest) -> Req<Nccl> {
  nccl::Impl::send(h.space(), sv, dest, h.get_inner());
  return Req<Nccl>(h.space.cuda_stream());
}

template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::Cuda, CommunicationSpace CommSpace = Nccl>
auto recv(const Handle<ExecSpace, CommSpace>& h, const RecvView& sv, int dest) -> Req<Nccl> {
  nccl::Impl::recv(h.space(), sv, dest, h.get_inner());
  return Req<Nccl>(h.space.cuda_stream());
}

} // namespace Experimental

} // namespace KokkosComm
