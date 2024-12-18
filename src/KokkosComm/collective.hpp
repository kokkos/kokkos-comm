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

#include <KokkosComm/fwd.hpp>

#include <Kokkos_Core_fwd.hpp>

#include <utility>

namespace KokkosComm {

template <KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
void barrier(Handle<ExecSpace, CommSpace>&& h) {
  Impl::Barrier<ExecSpace, CommSpace>{std::forward<Handle<ExecSpace, CommSpace>>(h)};
}

namespace Experimental {

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
auto reduce(Handle<ExecSpace, CommSpace>& h, const SendView& sv, RecvView& rv, int root) -> Req<CommSpace> {
  return Impl::Reduce<SendView, RecvView, RedOp, ExecSpace, CommSpace>::execute(h, sv, rv, root);
}

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
auto reduce(const SendView& sv, RecvView& rv, int root) -> Req<CommSpace> {
  return reduce(Handle<ExecSpace, CommSpace>{}, sv, rv, root);
}

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto allgather(Handle<ExecSpace, CommSpace>& h, const SendView& sv, RecvView& rv) -> Req<CommSpace> {
  return Impl::AllGather<SendView, RecvView, ExecSpace, CommSpace>::execute(h, sv, rv);
}

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto allgather(const SendView& sv, RecvView& rv) -> Req<CommSpace> {
  return allgather(Handle<ExecSpace, CommSpace>{}, sv, rv);
}

}  // namespace Experimental

}  // namespace KokkosComm
