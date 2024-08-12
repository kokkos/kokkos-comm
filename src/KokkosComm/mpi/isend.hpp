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

#include "KokkosComm/traits.hpp"

#include "mpi.hpp"
#include "impl/types.hpp"
#include "impl/tags.hpp"
#include "commmode.hpp"

namespace KokkosComm {

namespace Impl {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, mpi::CommunicationMode SendMode>
Req<Mpi> isend_impl(Handle<ExecSpace, Mpi> &h, const SendView &sv, int dest, int tag, SendMode) {
  auto mpi_isend_fn = [](void *mpi_view, int mpi_count, MPI_Datatype mpi_datatype, int mpi_dest, int mpi_tag,
                         MPI_Comm mpi_comm, MPI_Request *mpi_req) {
    if constexpr (std::is_same_v<SendMode, mpi::CommModeStandard>) {
      MPI_Isend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm, mpi_req);
    } else if constexpr (std::is_same_v<SendMode, mpi::CommModeReady>) {
      MPI_Irsend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm, mpi_req);
    } else if constexpr (std::is_same_v<SendMode, mpi::CommModeSynchronous>) {
      MPI_Issend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm, mpi_req);
    } else {
      static_assert(std::is_void_v<SendMode>, "unexpected communication mode");
    }
  };

  Req<Mpi> req;
  if (KokkosComm::is_contiguous(sv)) {
    h.space().fence("fence before isend");
    mpi_isend_fn(KokkosComm::data_handle(sv), 1, view_mpi_type(sv), dest, tag, h.mpi_comm(), &req.mpi_request());
    req.extend_view_lifetime(sv);
  } else {
    using Packer = typename KokkosComm::PackTraits<SendView>::packer_type;
    using Args   = typename Packer::args_type;

    Args args = Packer::pack(h.space(), sv);
    h.space().fence("fence before isend");
    mpi_isend_fn(args.view.data(), args.count, args.datatype, dest, tag, h.mpi_comm(), &req.mpi_request());
    req.extend_view_lifetime(args.view);
    req.extend_view_lifetime(sv);
  }
  return req;
}

// Implementation of KokkosComm::Send
template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
struct Send<SendView, ExecSpace, Mpi> {
  static Req<Mpi> execute(Handle<ExecSpace, Mpi> &h, const SendView &sv, int dest) {
    return isend_impl<ExecSpace, SendView>(h, sv, dest, POINTTOPOINT_TAG, mpi::DefaultCommMode{});
  }
};

}  // namespace Impl

namespace mpi {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, CommunicationMode SendMode>
Req<Mpi> isend(Handle<ExecSpace, Mpi> &h, const SendView &sv, int dest, int tag, SendMode) {
  return KokkosComm::Impl::isend_impl<ExecSpace, SendView>(h, sv, dest, tag, SendMode{});
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
Req<Mpi> isend(Handle<ExecSpace, Mpi> &h, const SendView &sv, int dest, int tag) {
  return isend<ExecSpace, SendView>(h, sv, dest, tag, DefaultCommMode{});
}

template <KokkosView SendView>
void isend(const SendView &sv, int dest, int tag, MPI_Comm comm, MPI_Request &req) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::isend");

  if (KokkosComm::is_contiguous(sv)) {
    using SendScalar = typename SendView::non_const_value_type;
    MPI_Isend(KokkosComm::data_handle(sv), KokkosComm::span(sv), Impl::mpi_type_v<SendScalar>, dest, tag, comm, &req);
  } else {
    throw std::runtime_error("only contiguous views supported for low-level isend");
  }
  Kokkos::Tools::popRegion();
}

}  // namespace mpi

}  // namespace KokkosComm