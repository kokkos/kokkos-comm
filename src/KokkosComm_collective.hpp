#pragma once

#include "impl/KokkosComm_reduce.hpp"

#include <Kokkos_Core.hpp>

namespace KokkosComm {

template <typename SendView, typename RecvView, typename ExecSpace>
void reduce(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Op op, int root,
          MPI_Comm comm) {
  return Impl::reduce(space, sv, rv, op, root, comm);
}

} // namespace KokkosComm
