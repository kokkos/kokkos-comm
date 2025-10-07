// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>

#include "impl/types.hpp"

namespace KokkosComm::mpi {

template <KokkosView View>
void broadcast(View const& v, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::broadcast");

  using Scalar = typename View::value_type;

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(v), "low-level broadcast requires contiguous view");

  MPI_Bcast(KokkosComm::data_handle(v), KokkosComm::span(v), KokkosComm::Impl::mpi_type_v<Scalar>, root, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView View>
void broadcast(ExecSpace const& space, View const& v, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::broadcast");

  space.fence("fence before broadcast");  // work in space may have been used to produce view data
  broadcast(v, root, comm);

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::mpi
