// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <KokkosComm/concepts.hpp>

namespace KokkosComm::mpi {

inline void barrier(MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::barrier");
  MPI_Barrier(comm);
  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::mpi
