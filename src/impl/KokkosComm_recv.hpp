#pragma once

#include <mpi.h>

#include <Kokkos_Core.hpp>

// impl
#include "KokkosComm_unpack.hpp"

namespace KokkosComm::Impl {
template <typename RecvView, typename ExecSpace>
void recv(const ExecSpace &space, const RecvView &rv, int src, int tag,
          MPI_Comm comm) {

  using value_type = typename RecvView::value_type;
  using non_const_value_type = typename RecvView::non_const_value_type;

  if (rv.span_is_contiguous()) {
    MPI_Recv(rv.data(), rv.span() * sizeof(value_type), MPI_PACKED, src, tag,
             comm, MPI_STATUS_IGNORE);
  } else {
    if constexpr (RecvView::rank == 1) {
      Kokkos::View<typename RecvView::non_const_value_type *> packed(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "packed"),
          rv.extent(0));
      MPI_Recv(packed.data(), packed.span() * sizeof(value_type), MPI_PACKED,
               src, tag, comm, MPI_STATUS_IGNORE);
      unpack(space, rv, packed);
    } else if constexpr (RecvView::rank == 2) {
      Kokkos::View<typename RecvView::non_const_value_type **> packed(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "packed"),
          rv.extent(0), rv.extent(1));
      MPI_Recv(packed.data(), packed.span() * sizeof(value_type), MPI_PACKED,
               src, tag, comm, MPI_STATUS_IGNORE);
      unpack(space, rv, packed);
    } else {
      static_assert(std::is_void_v<RecvView>,
                    "recv only supports rank-1 views");
    }
  }
}
} // namespace KokkosComm::Impl