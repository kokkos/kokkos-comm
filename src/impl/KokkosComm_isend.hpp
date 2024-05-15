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

#include <iostream>

#include <Kokkos_Core.hpp>

#include "KokkosComm_concepts.hpp"
#include "KokkosComm_pack_traits.hpp"
#include "KokkosComm_request.hpp"
#include "KokkosComm_traits.hpp"
#include "KokkosComm_comm_mode.hpp"

// impl
#include "KokkosComm_include_mpi.hpp"

namespace KokkosComm::Impl {

template <CommMode SendMode = CommMode::Default, KokkosExecutionSpace ExecSpace, KokkosView SendView>
KokkosComm::Req isend(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::isend");

  KokkosComm::Req req;

  using KCT  = KokkosComm::Traits<SendView>;
  using KCPT = KokkosComm::PackTraits<SendView>;

  auto mpi_isend_fn = [](void *mpi_view, int mpi_count, MPI_Datatype mpi_datatype, int mpi_dest, int mpi_tag,
                         MPI_Comm mpi_comm, MPI_Request *mpi_req) {
    if constexpr (SendMode == CommMode::Standard) {
      MPI_Isend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm, mpi_req);
    } else if constexpr (SendMode == CommMode::Buffered) {
      MPI_Ibsend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm, mpi_req);
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

  if (KCPT::needs_pack(sv)) {
    using Packer  = typename KCPT::packer_type;
    using MpiArgs = typename Packer::args_type;

    MpiArgs args = Packer::pack(space, sv);
    space.fence();
    mpi_isend_fn(KCT::data_handle(args.view), args.count, args.datatype, dest, tag, comm, &req.mpi_req());
    req.keep_until_wait(args.view);
  } else {
    using SendScalar = typename SendView::value_type;
    mpi_isend_fn(KCT::data_handle(sv), KCT::span(sv), mpi_type_v<SendScalar>, dest, tag, comm, &req.mpi_req());
    if (KCT::is_reference_counted()) {
      req.keep_until_wait(sv);
    }
  }

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace KokkosComm::Impl
