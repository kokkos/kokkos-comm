// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "mpi_space.hpp"
#include "handle.hpp"
#include "req.hpp"

#include "impl/pack_traits.hpp"
#include "impl/tags.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace Impl {

// Recv implementation for Mpi
template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
struct Recv<RecvView, ExecSpace, MpiSpace> {
  static Req<MpiSpace> execute(Handle<ExecSpace, MpiSpace> &h, const RecvView &rv, int src) {
    using Packer = typename mpi::Impl::PackTraits<RecvView>::packer_type;

    const ExecSpace &space = h.space();

    Req<MpiSpace> req;
    if (KokkosComm::is_contiguous(rv)) {
      space.fence("fence before irecv");
      MPI_Irecv(KokkosComm::data_handle(rv), KokkosComm::span(rv), datatype<MpiSpace, typename RecvView::value_type>(),
                src, POINTTOPOINT_TAG, h.mpi_comm(), &req.mpi_request());
      req.extend_view_lifetime(rv);
    } else {
      auto args = Packer::allocate_packed_for(space, "TODO", rv);
      space.fence("fence before irecv");
      MPI_Irecv(args.view.data(), args.count, args.datatype, src, POINTTOPOINT_TAG, h.mpi_comm(), &req.mpi_request());
      // implicitly extends args.view and rv lifetime due to lambda capture
      req.call_after_mpi_wait([=]() { Packer::unpack_into(space, rv, args.view); });
    }
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
