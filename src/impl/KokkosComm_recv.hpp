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

#include "KokkosComm_concepts.hpp"
#include "KokkosComm_traits.hpp"

// impl
#include "KokkosComm_include_mpi.hpp"

namespace KokkosComm::Impl {
template <KokkosExecutionSpace ExecSpace, KokkosView RecvView,
          NonContigSendRecv NC = DefaultNonContigSendRecv<ExecSpace, RecvView>>
void recv(const ExecSpace &space, RecvView &rv, int src, int tag, MPI_Comm comm) {
  using KCT = KokkosComm::Traits<RecvView>;
  Kokkos::Tools::pushRegion("KokkosComm::Impl::recv");

  // I think it's okay to use the same tag for all messages here due to
  // non-overtaking of messages that match the same recv
  CtxBufCount ctx = NC::pre_recv(space, rv);
  space.fence();  // space may be allocating our view
  for (const CtxBufCount::MpiArgs &args : ctx.mpi_args) {
    MPI_Recv(args.buf, args.count, args.datatype, src, tag, comm, MPI_STATUS_IGNORE);
  }
  NC::post_recv(space, rv, ctx);

  Kokkos::Tools::popRegion();
}
}  // namespace KokkosComm::Impl
