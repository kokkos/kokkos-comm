#pragma once

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include "KokkosComm_pack.hpp"

// impl
#include "KokkosComm_pack.hpp"

namespace KokkosComm::Impl {
template <typename SendView, typename ExecSpace>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag,
          MPI_Comm comm) {
  if (KokkosComm::Traits<SendView>::needs_pack(sv)) {
    if constexpr (SendView::rank == 1) {
      using SendScalar = typename KokkosComm::Traits<
          SendView>::non_const_packed_view_type::value_type;
      auto packed = allocate_packed_for(space, "packed", sv);
      pack(space, packed, sv);
      space.fence();
      MPI_Send(packed.data(), packed.span() * sizeof(SendScalar), MPI_PACKED,
               dest, tag, comm);
    } else {
      static_assert(std::is_void_v<SendView>,
                    "send only supports rank-1 views");
    }
  } else {
    using SendScalar = typename SendView::value_type;
    MPI_Send(sv.data(), sv.span(), mpi_type_v<SendScalar>, dest, tag, comm);
  }
}
} // namespace KokkosComm::Impl