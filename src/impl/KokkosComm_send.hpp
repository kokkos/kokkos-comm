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

  using value_type = typename SendView::value_type;

  if (sv.span_is_contiguous()) {
    MPI_Send(sv.data(), sv.span() * sizeof(value_type), MPI_PACKED, dest, tag,
             comm);
  } else {
    if constexpr (SendView::rank == 1) {
      Kokkos::View<typename SendView::non_const_value_type *> packed(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "packed"),
          sv.extent(0));
      pack(space, packed, sv);
      space.fence();
      MPI_Send(packed.data(), packed.span() * sizeof(value_type), MPI_PACKED,
               dest, tag, comm);
    } else {
      static_assert(std::is_void_v<SendView>,
                    "send only supports rank-1 views");
    }
  }
}
} // namespace KokkosComm::Impl