// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <KokkosComm/concepts.hpp>

namespace KokkosComm {
namespace Impl {

template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
struct Barrier {
  Barrier(Handle<ExecSpace, Mpi> &&h) {
    h.space().fence("KokkosComm::Impl::Barrier");
    MPI_Barrier(h.mpi_comm());
  }
};

}  // namespace Impl
namespace mpi {

inline void barrier(MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::barrier");
  MPI_Barrier(comm);
  Kokkos::Tools::popRegion();
}

}  // namespace mpi
}  // namespace KokkosComm
