// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include "mpi_space.hpp"
#include "comm_mode.hpp"
#include "handle.hpp"

#include "impl/pack_traits.hpp"
#include "impl/tags.hpp"
#include "impl/types.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace Impl {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, mpi::CommunicationMode SendMode>
Req<MpiSpace> isend_impl(Handle<ExecSpace, MpiSpace> &h, const SendView &sv, int dest, int tag, SendMode) {
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

  Req<MpiSpace> req;
  if (KokkosComm::is_contiguous(sv)) {
    h.space().fence("fence before isend");
    mpi_isend_fn(KokkosComm::data_handle(sv), KokkosComm::span(sv), mpi_type_v<typename SendView::value_type>, dest,
                 tag, h.mpi_comm(), &req.mpi_request());
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
struct Send<SendView, ExecSpace, MpiSpace> {
  static Req<MpiSpace> execute(Handle<ExecSpace, MpiSpace> &h, const SendView &sv, int dest) {
    return isend_impl<ExecSpace, SendView>(h, sv, dest, POINTTOPOINT_TAG, mpi::DefaultCommMode{});
  }
};

}  // namespace Impl
namespace mpi {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, CommunicationMode SendMode>
Req<MpiSpace> isend(Handle<ExecSpace, MpiSpace> &h, const SendView &sv, int dest, int tag, SendMode) {
  return KokkosComm::Impl::isend_impl<ExecSpace, SendView>(h, sv, dest, tag, SendMode{});
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
Req<MpiSpace> isend(Handle<ExecSpace, MpiSpace> &h, const SendView &sv, int dest, int tag) {
  return isend<ExecSpace, SendView>(h, sv, dest, tag, DefaultCommMode{});
}

template <KokkosView SendView>
void isend(const SendView &sv, int dest, int tag, MPI_Comm comm, MPI_Request &req) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::isend");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(sv), "only contiguous views supported for low-level isend");

  using SendScalar = typename SendView::non_const_value_type;
  MPI_Isend(KokkosComm::data_handle(sv), KokkosComm::span(sv), Impl::mpi_type_v<SendScalar>, dest, tag, comm, &req);

  Kokkos::Tools::popRegion();
}

}  // namespace mpi
}  // namespace KokkosComm
