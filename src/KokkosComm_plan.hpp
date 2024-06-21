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

#include "KokkosComm_fwd.hpp"

namespace KokkosComm {

template <Dispatch DISPATCH, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
Handle<ExecSpace, TRANSPORT> plan(const ExecSpace &space, MPI_Comm comm, DISPATCH d) {
  return Plan<DISPATCH, ExecSpace, TRANSPORT>(space, comm, d).handle();
}

template <Dispatch DISPATCH, KokkosExecutionSpace ExecSpace, Transport TRANSPORT>
void plan(Handle<ExecSpace, TRANSPORT> &handle, DISPATCH d) {
  Plan<DISPATCH, ExecSpace, TRANSPORT>(handle, d);
}

}  // namespace KokkosComm
