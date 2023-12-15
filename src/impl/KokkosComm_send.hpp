#pragma once

#include <mpi.h>

#include "KokkosComm_pack.hpp"

namespace KokkosComm::Impl {
template <typename SendView, typename ExecSpace>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag,
          MPI_Comm comm) {

  using value_type = typename SendView::non_const_value_type;

  auto packedView = Impl::pack(space, sv, comm);

  MPI_Send(packedView.data(), packedView.size(), MPI_PACKED, dest, tag, comm);
}
} // namespace KokkosComm::Impl