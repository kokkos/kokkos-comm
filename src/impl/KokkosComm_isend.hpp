#pragma once

#include <iostream>

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include "KokkosComm_request.hpp"

// impl
#include "KokkosComm_pack.hpp"

namespace KokkosComm::Impl {
template <typename SendView, typename ExecSpace>
KokkosComm::Req isend(const ExecSpace &space, const SendView &sv, int dest,
                      int tag, MPI_Comm comm) {
  KokkosComm::Req req;

  if (KokkosComm::Traits<SendView>::needs_pack(sv)) {
    using PackedScalar =
        typename KokkosComm::Traits<SendView>::packed_view_type::value_type;
    auto packed = allocate_packed_for(space, "packed", sv);
    pack(space, packed, sv);
    space.fence();
    MPI_Isend(packed.data(), packed.span() * sizeof(PackedScalar), MPI_PACKED,
              dest, tag, comm, &req.mpi_req());
    req.keep_until_wait(packed);
    return req;
  } else {
    using SendScalar = typename SendView::value_type;
    MPI_Isend(sv.data(), sv.span(), mpi_type_v<SendScalar>, dest, tag, comm,
              &req.mpi_req());
    req.keep_until_wait(sv);
    return req;
  }
}
} // namespace KokkosComm::Impl