// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "mpi_space.hpp"
#include "handle.hpp"

#include <KokkosComm/impl/host_staging.hpp>
#include "impl/pack_traits.hpp"
#include "impl/tags.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace Impl {

// Recv implementation for Mpi
template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
struct Recv<RecvView, ExecSpace, MpiSpace> {
  static Req<MpiSpace> execute(Handle<ExecSpace, MpiSpace> &h, const RecvView &rv, int src) {
    using T      = typename RecvView::non_const_value_type;
    using Packer = typename KokkosComm::PackTraits<RecvView>::packer_type;

    const ExecSpace &space = h.space();

    Req<MpiSpace> req;
#if defined(KOKKOSCOMM_ENABLE_GPU_AWARE_MPI)
    if (is_contiguous(rv)) {
      space.fence("fence before GPU-aware `MPI_Irecv`");
      MPI_Irecv(data_handle(rv), span(rv), datatype<MpiSpace, T>(), src, POINTTOPOINT_TAG, h.mpi_comm(),
                &req.mpi_request());
    } else {
      auto args = Packer::allocate_packed_for(space, "TODO", rv);
      space.fence("fence packing before GPU-aware `MPI_Irecv`");
      MPI_Irecv(args.view.data(), args.count, args.datatype, src, POINTTOPOINT_TAG, h.mpi_comm(), &req.mpi_request());
      // Implicitly extends args.view and rv lifetime due to lambda capture
      req.call_after_mpi_wait([=]() { Packer::unpack_into(space, rv, args.view); });
    }
    req.extend_view_lifetime(rv);
#else
    auto host_rv = KokkosComm::Impl::stage_for(rv);
    space.fence("fence host staging before `MPI_Irecv`");
    if (is_contiguous(host_rv)) {
      MPI_Irecv(data_handle(host_rv), span(host_rv), datatype<MpiSpace, T>(), src, POINTTOPOINT_TAG, h.mpi_comm(),
                &req.mpi_request());
      req.extend_view_lifetime(host_rv);
    } else {
      auto args = Packer::allocate_packed_for(space, "packed `MPI_Irecv`", host_rv);
      space.fence("fence packing before `MPI_Irecv`");
      MPI_Irecv(args.view.data(), args.count, args.datatype, src, POINTTOPOINT_TAG, h.mpi_comm(), &req.mpi_request());
      // Implicitly extends `args.view`, `host_rv` and `rv` lifetimes due to lambda capture
      // TODO: Can we unpack directly into `rv` instead of `host_rv`?
      req.call_after_mpi_wait([=]() { Packer::unpack_into(space, rv, args.view); });
      req.call_after_mpi_wait([=]() { KokkosComm::Impl::copy_back(space, rv, host_rv); });
    }
    req.extend_view_lifetime(rv);
#endif
    return req;
  }
};

}  // namespace Impl
namespace mpi {

template <KokkosView RecvView>
void irecv(const RecvView &rv, int src, int tag, MPI_Comm comm, MPI_Request &req) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::irecv");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(rv), "Only contiguous irecv viewsupported");

  using RecvScalar = typename RecvView::non_const_value_type;
  MPI_Irecv(KokkosComm::data_handle(rv), KokkosComm::span(rv), datatype<MpiSpace, RecvScalar>(), src, tag, comm, &req);

  Kokkos::Tools::popRegion();
}

}  // namespace mpi
}  // namespace KokkosComm
