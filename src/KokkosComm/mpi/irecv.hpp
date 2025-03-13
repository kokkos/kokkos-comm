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

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include "mpi_space.hpp"
#include "handle.hpp"

#include "impl/pack_traits.hpp"
#include "impl/tags.hpp"
#include "impl/types.hpp"

namespace KokkosComm {
namespace Impl {

// Recv implementation for Mpi
template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
struct Recv<RecvView, ExecSpace, Mpi> {
  static Req<Mpi> execute(Handle<ExecSpace, Mpi> &h, const RecvView &rv, int src) {
    using KCPT   = KokkosComm::PackTraits<RecvView>;
    using Packer = typename KCPT::packer_type;
    using Args   = typename Packer::args_type;

    const ExecSpace &space = h.space();

    Req<Mpi> req;
    if (KokkosComm::is_contiguous(rv)) {
      space.fence("fence before irecv");
      MPI_Irecv(KokkosComm::data_handle(rv), KokkosComm::span(rv), mpi_type_v<typename RecvView::value_type>, src,
                POINTTOPOINT_TAG, h.mpi_comm(), &req.mpi_request());
      req.extend_view_lifetime(rv);
    } else {
      Args args = Packer::allocate_packed_for(space, "TODO", rv);
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

  if (KokkosComm::is_contiguous(rv)) {
    using RecvScalar = typename RecvView::non_const_value_type;
    MPI_Irecv(KokkosComm::data_handle(rv), KokkosComm::span(rv), Impl::mpi_type_v<RecvScalar>, src, tag, comm, &req);
  } else {
    throw std::runtime_error("Only contiguous irecv viewsupported");
  }
  Kokkos::Tools::popRegion();
}

}  // namespace mpi

}  // namespace KokkosComm
