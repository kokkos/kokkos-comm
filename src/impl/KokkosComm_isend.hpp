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
#include "KokkosComm_request.hpp"
#include "KokkosComm_traits.hpp"
#include "KokkosComm_comm_mode.hpp"

#include "impl/KokkosComm_include_mpi.hpp"

namespace KokkosComm::Impl {

template <CommMode SendMode = CommMode::Default, KokkosExecutionSpace ExecSpace, KokkosView SendView,
          NonContig NC      = DefaultNonContig<ExecSpace, SendView>>
Req isend(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::isend");
  Req req;

  auto mpi_isend_fn = [](void *mpi_view, int mpi_count, MPI_Datatype mpi_datatype, int mpi_dest, int mpi_tag,
                         MPI_Comm mpi_comm, MPI_Request *mpi_req) {
    if constexpr (SendMode == CommMode::Standard) {
      MPI_Isend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm, mpi_req);
    } else if constexpr (SendMode == CommMode::Ready) {
      MPI_Irsend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm, mpi_req);
    } else if constexpr (SendMode == CommMode::Synchronous) {
      MPI_Issend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm, mpi_req);
    } else if constexpr (SendMode == CommMode::Default) {
#ifdef KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE
      MPI_Issend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm, mpi_req);
#else
      MPI_Isend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm, mpi_req);
#endif
    }
  };

  // I think it's okay to use the same tag for all messages here due to
  // non-overtaking of messages that match the same recv
  Ctx ctx = NC::pre_send(space, sv);  // FIXME: terrible name
  space.fence();
  for (Ctx::MpiArgs &args : ctx.mpi_args) {
    mpi_isend_fn(args.buf, args.count, args.datatype, dest, tag, comm, &args.req);
  }
  for (auto v : ctx.wait_callbacks) {
    req.call_and_drop_at_wait(v);
  }

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace KokkosComm::Impl
