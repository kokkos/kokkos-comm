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

#include "KokkosComm_api.hpp"
#include "KokkosComm_concepts.hpp"
#include "KokkosComm_config.hpp"
#include "KokkosComm_plan.hpp"

namespace KokkosComm {

template <typename Handle, KokkosView RecvView>
void irecv(Handle &h, RecvView &rv, int src, int tag) {
  if constexpr (Impl::api_avail_v<SpecialTransport, Impl::Api::Irecv>) {
    SpecialTransport::irecv(h, rv, src, tag);
  } else {
    GenericTransport::irecv(h, rv, src, tag);
  }
}

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
KokkosComm::Handle<ExecSpace> irecv(const ExecSpace &space, const RecvView &rv, int dest, int tag, MPI_Comm comm) {
  using MyHandle = KokkosComm::Handle<ExecSpace>;
  return KokkosComm::plan(space, comm, [=](MyHandle &handle) { KokkosComm::irecv(handle, rv, dest, tag); });
}

template <typename Handle, KokkosView SendView>
void isend(Handle &h, SendView &sv, int src, int tag) {
  if constexpr (Impl::api_avail_v<SpecialTransport, Impl::Api::Isend>) {
    SpecialTransport::isend(h, sv, src, tag);
  } else {
    GenericTransport::isend(h, sv, src, tag);
  }
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
KokkosComm::Handle<ExecSpace> isend(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  using MyHandle = KokkosComm::Handle<ExecSpace>;
  return KokkosComm::plan(space, comm, [=](MyHandle &handle) { KokkosComm::isend(handle, sv, dest, tag); });
}

}  // namespace KokkosComm
