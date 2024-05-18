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

#include <Kokkos_Core.hpp>

#include "KokkosComm_mpi.hpp"
#include "KokkosComm_pack_traits.hpp"
#include "KokkosComm_traits.hpp"

namespace KokkosComm::Impl {

inline void barrier(Communicator comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::barrier");
  comm.barrier();
  Kokkos::Tools::popRegion();
}

// a barrier in the provided space. For MPI, we have to fence the space and do a host barrier
template <KokkosExecutionSpace ExecSpace>
void barrier(const ExecSpace &space, Communicator comm) {
  space.fence("KokkosComm::Impl::barrier");
  barrier(comm);
}
}  // namespace KokkosComm::Impl
