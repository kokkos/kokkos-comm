#pragma once

#include <mpi.h>

#include "KokkosComm_packed_size.hpp"
#include "KokkosComm_unpack.hpp"

namespace KokkosComm::Impl {
template <typename RecvView, typename ExecSpace>
void recv(const ExecSpace &space, const RecvView &rv, int src, int tag,
          MPI_Comm comm) {
  using PackedView = Kokkos::View<typename RecvView::non_const_value_type *>;
  size_t packedSize = Impl::packed_size(rv);
  PackedView packedView(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "packed"),
      packedSize);
  MPI_Recv(packedView.data(), packedView.size(), MPI_PACKED, src, tag, comm,
           MPI_STATUS_IGNORE);
  Impl::unpack_into(space, rv, packedView);
}
} // namespace KokkosComm::Impl