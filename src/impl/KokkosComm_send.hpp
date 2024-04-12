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

#include "KokkosComm_pack_traits.hpp"
#include "KokkosComm_comm_mode.hpp"

// impl
#include "KokkosComm_include_mpi.hpp"

namespace KokkosComm::Impl {

template <KokkosView SendView>
void send(const SendView &sv, int dest, int tag, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::send");
  using KCT = typename KokkosComm::Traits<SendView>;

  if (KCT::is_contiguous(sv)) {
    using SendScalar = typename SendView::non_const_value_type;
    MPI_Send(KCT::data_handle(sv), KCT::span(sv), mpi_type_v<SendScalar>, dest, tag, comm);
  } else {
    throw std::runtime_error("only contiguous views supported for low-level send");
  }
  Kokkos::Tools::popRegion();
}

template <CommMode SendMode = CommMode::Default, KokkosExecutionSpace ExecSpace, KokkosView SendView,
          NonContig NC      = DefaultNonContig<ExecSpace, SendView>>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::send");
  Req req;

  space.fence("send fence before checking view properties");

  auto mpi_send_fn = [](void *mpi_view, int mpi_count, MPI_Datatype mpi_datatype, int mpi_dest, int mpi_tag,
                        MPI_Comm mpi_comm) {
    if constexpr (SendMode == CommMode::Standard) {
      MPI_Send(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm);
    } else if constexpr (SendMode == CommMode::Ready) {
      MPI_Rsend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm);
    } else if constexpr (SendMode == CommMode::Synchronous) {
      MPI_Ssend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm);
    } else if constexpr (SendMode == CommMode::Default) {
#ifdef KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE
      MPI_Ssend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm);
#else
      MPI_Send(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm);
#endif
    }
  };

  // I think it's okay to use the same tag for all messages here due to
  // non-overtaking of messages that match the same recv
  const Ctx ctx = NC::pre_send(space, sv);  // FIXME: terrible name
  space.fence();
  for (const Ctx::MpiArgs &args : ctx.mpi_args) {
    mpi_send_fn(args.buf, args.count, args.datatype, dest, tag, comm);
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::Impl
