#pragma once

#include "impl/KokkosComm_send.hpp"

#include "Kokkos.hpp"

namespace KokkosComm {
template <typename SendView, typename ExecSpace>
void send(const SendView &sv, int dest, int tag, MPI_Comm comm);
} // namespace KokkosComm
