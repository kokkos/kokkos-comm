// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "mpi_space.hpp"
#include "req.hpp"

#include "impl/error_handling.hpp"

namespace KokkosComm::mpi {

template <KokkosExecutionSpace ExecSpace, KokkosView View>
auto ibroadcast(const ExecSpace& space, View& v, int root, MPI_Comm comm) -> Req<MpiSpace> {
  using T = typename View::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::mpi::ibroadcast");
  fail_if(!is_contiguous(v), "KokkosComm::mpi::ibroadcast: unimplemented for non-contiguous views");

  // Sync: Work in space may have been used to produce view data.
  space.fence("fence before non-blocking broadcast");

  Req<MpiSpace> req;
  MPI_Ibcast(data_handle(v), span(v), datatype<MpiSpace, T>, root, comm, &req.mpi_request());
  req.extend_view_lifetime(v);

  Kokkos::Tools::popRegion();
  return req;
}

template <KokkosView View>
void broadcast(View const& v, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::broadcast");

  using Scalar = typename View::value_type;

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(v), "low-level broadcast requires contiguous view");

  MPI_Bcast(KokkosComm::data_handle(v), KokkosComm::span(v), datatype<MpiSpace, Scalar>(), root, comm);

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
