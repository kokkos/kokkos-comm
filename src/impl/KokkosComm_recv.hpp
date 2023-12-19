#pragma once

#include <mpi.h>

#include <Kokkos_Core.hpp>

// impl
#include "KokkosComm_unpack.hpp"

namespace KokkosComm::Impl {
template <typename RecvView, typename ExecSpace>
void recv(const ExecSpace &space, const RecvView &rv, int src, int tag,
          MPI_Comm comm) {
  if (KokkosComm::Traits<RecvView>::needs_unpack(rv)) {
    using PackedScalar =
        typename KokkosComm::Traits<RecvView>::packed_view_type::value_type;
    auto packed = allocate_packed_for(space, "packed", rv);
    space.fence();
    MPI_Recv(packed.data(), packed.span() * sizeof(PackedScalar), MPI_PACKED,
             src, tag, comm, MPI_STATUS_IGNORE);
    unpack(space, rv, packed);
    space.fence();
  } else {
    using RecvScalar = typename RecvView::value_type;
    MPI_Recv(rv.data(), rv.span(), mpi_type_v<RecvScalar>, src, tag, comm,
             MPI_STATUS_IGNORE);
  }
}
} // namespace KokkosComm::Impl