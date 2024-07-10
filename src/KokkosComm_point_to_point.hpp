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
Req<TRANSPORT> irecv(Handle<ExecSpace, TRANSPORT> &h, RecvView &rv, int src, int tag) {
  return Impl::Irecv<RecvView, ExecSpace, TRANSPORT>::execute(h, rv, src, tag);
}

template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
Req<TRANSPORT> irecv(RecvView &rv, int src, int tag) {
  return irecv<RecvView, ExecSpace, TRANSPORT>(Handle<ExecSpace, TRANSPORT>{}, rv, src, tag);
}

template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
Req<TRANSPORT> isend(Handle<ExecSpace, TRANSPORT> &h, SendView &sv, int dest, int tag) {
  return Impl::Isend<SendView, ExecSpace, TRANSPORT>::execute(h, sv, dest, tag);
}

template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
Req<TRANSPORT> isend(SendView &sv, int dest, int tag) {
  return isend<SendView, ExecSpace, TRANSPORT>(Handle<ExecSpace, TRANSPORT>{}, sv, dest, tag);
}

}  // namespace KokkosComm
