#pragma once

#include <mpi.h>

#include "KokkosComm_pack.hpp"

namespace KokkosComm::Impl {
template <typename SendView, typename ExecSpace>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {

    using value_type = SendView::non_const_value_type;

    KokkosView<char *> packed = Impl::pack(space, sv, comm);

    MPI_Send(packed.data(), packed.size(), MPI_PACKED, dest, tag, comm);
}
} // namespace KokkosComm