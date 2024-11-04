//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

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
