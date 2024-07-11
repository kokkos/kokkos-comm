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

namespace KokkosComm {

template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
Req<TRANSPORT> recv(Handle<ExecSpace, TRANSPORT> &h, RecvView &rv, int src, int tag) {
  return Impl::Recv<RecvView, ExecSpace, TRANSPORT>::execute(h, rv, src, tag);
}

template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
Req<TRANSPORT> recv(RecvView &rv, int src, int tag) {
  return recv<RecvView, ExecSpace, TRANSPORT>(Handle<ExecSpace, TRANSPORT>{}, rv, src, tag);
}

template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
Req<TRANSPORT> send(Handle<ExecSpace, TRANSPORT> &h, SendView &sv, int dest, int tag) {
  return Impl::Send<SendView, ExecSpace, TRANSPORT>::execute(h, sv, dest, tag);
}

template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
Req<TRANSPORT> send(SendView &sv, int dest, int tag) {
  return send<SendView, ExecSpace, TRANSPORT>(Handle<ExecSpace, TRANSPORT>{}, sv, dest, tag);
}

}  // namespace KokkosComm
