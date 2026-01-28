// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "KokkosComm/impl/host_staging.hpp"
#include "mpi_space.hpp"
#include "comm_mode.hpp"
#include "handle.hpp"

#include <KokkosComm/impl/host_staging.hpp>
#include "impl/pack_traits.hpp"
#include "impl/tags.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace Impl {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, mpi::CommunicationMode SendMode>
Req<MpiSpace> isend_impl(Handle<ExecSpace, MpiSpace>& h, const SendView& sv, int dest, int tag, SendMode) {
  using T = typename SendView::non_const_value_type;

  auto mpi_isend_fn = [](void* mpi_view, int mpi_count, MPI_Datatype mpi_datatype, int mpi_dest, int mpi_tag,
                         MPI_Comm mpi_comm, MPI_Request* mpi_req) {
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
#if defined(KOKKOSCOMM_ENABLE_GPU_AWARE_MPI)
  if (is_contiguous(sv)) {
    h.space().fence("fence before GPU-aware `MPI_Isend`");
    mpi_isend_fn(data_handle(sv), span(sv), datatype<MpiSpace, T>(), dest, tag, h.mpi_comm(), &req.mpi_request());
  } else {
    using Packer = typename KokkosComm::PackTraits<SendView>::packer_type;
    auto args    = Packer::pack(h.space(), sv);
    h.space().fence("fence packing before GPU-aware `MPI_Isend`");
    mpi_isend_fn(data_handle(args.view), args.count, args.datatype, dest, tag, h.mpi_comm(), &req.mpi_request());
    req.extend_view_lifetime(args.view);
  }
  req.extend_view_lifetime(sv);
#else
  auto host_sv = KokkosComm::Impl::stage_for(sv);
  h.space().fence("fence host staging before `MPI_Isend`");
  if (is_contiguous(host_sv)) {
    mpi_isend_fn(data_handle(host_sv), span(host_sv), datatype<MpiSpace, T>(), dest, tag, h.mpi_comm(),
                 &req.mpi_request());
  } else {
    using Packer = typename KokkosComm::PackTraits<decltype(host_sv)>::packer_type;
    auto args    = Packer::pack(h.space(), host_sv);
    h.space().fence("fence packing before `MPI_Isend`");
    mpi_isend_fn(data_handle(args.view), args.count, args.datatype, dest, tag, h.mpi_comm(), &req.mpi_request());
    req.extend_view_lifetime(args.view);
  }
  req.extend_view_lifetime(host_sv);
  // TODO: Do we need to extend the lifetime of `sv` if we are staging it on the host?
  req.extend_view_lifetime(sv);
#endif
  return req;
}

// Implementation of KokkosComm::Send
template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
struct Send<SendView, ExecSpace, MpiSpace> {
  static Req<MpiSpace> execute(Handle<ExecSpace, MpiSpace>& h, const SendView& sv, int dest) {
    return isend_impl<ExecSpace, SendView>(h, sv, dest, POINTTOPOINT_TAG, mpi::DefaultCommMode{});
  }
};

}  // namespace Impl
namespace mpi {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, CommunicationMode SendMode>
Req<MpiSpace> isend(Handle<ExecSpace, MpiSpace>& h, const SendView& sv, int dest, int tag, SendMode) {
  return KokkosComm::Impl::isend_impl<ExecSpace, SendView>(h, sv, dest, tag, SendMode{});
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
Req<MpiSpace> isend(Handle<ExecSpace, MpiSpace>& h, const SendView& sv, int dest, int tag) {
  return isend<ExecSpace, SendView>(h, sv, dest, tag, DefaultCommMode{});
}

template <KokkosView SendView>
void isend(const SendView& sv, int dest, int tag, MPI_Comm comm, MPI_Request& req) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::isend");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(sv), "only contiguous views supported for low-level isend");

  using SendScalar = typename SendView::non_const_value_type;
  MPI_Isend(KokkosComm::data_handle(sv), KokkosComm::span(sv), datatype<MpiSpace, SendScalar>(), dest, tag, comm, &req);

  Kokkos::Tools::popRegion();
}

}  // namespace mpi
}  // namespace KokkosComm
