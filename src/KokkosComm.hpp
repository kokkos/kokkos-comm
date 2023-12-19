#pragma once

#include "KokkosComm_collective.hpp"
#include "KokkosComm_version.hpp"
#include "impl/KokkosComm_isend.hpp"
#include "impl/KokkosComm_recv.hpp"
#include "impl/KokkosComm_send.hpp"

#include <Kokkos_Core.hpp>

namespace KokkosComm {

template <typename SendView, typename ExecSpace>
Req isend(const ExecSpace &space, const SendView &sv, int dest, int tag,
          MPI_Comm comm) {
  return Impl::isend(space, sv, dest, tag, comm);
}

template <typename SendView, typename ExecSpace>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag,
          MPI_Comm comm) {
  return Impl::send(space, sv, dest, tag, comm);
}

template <typename Recv, typename ExecSpace>
void recv(const ExecSpace &space, const Recv &sv, int src, int tag,
          MPI_Comm comm) {
  return Impl::recv(space, sv, src, tag, comm);
}

} // namespace KokkosComm
