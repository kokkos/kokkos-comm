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

#include <mpi.h>
#include "KokkosComm_collective.hpp"
#include "KokkosComm_communicator.hpp"
#include "KokkosComm_version.hpp"
#include "KokkosComm_isend.hpp"
#include "KokkosComm_recv.hpp"
#include "KokkosComm_send.hpp"
#include "KokkosComm_concepts.hpp"
#include "KokkosComm_comm_mode.hpp"
#include "KokkosComm_MPI_instance.hpp"

#include <Kokkos_Core.hpp>

namespace KokkosComm {

template <CommMode SendMode = CommMode::Default, KokkosExecutionSpace ExecSpace, KokkosView SendView>
Req isend(const ExecSpace &space, const SendView &sv, int dest, int tag, const Communicator<ExecSpace> &comm) {
  return Impl::isend<SendMode>(space, sv, dest, tag, comm.as_raw());
}

template <CommMode SendMode = CommMode::Default, KokkosExecutionSpace ExecSpace, KokkosView SendView>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, const Communicator<ExecSpace> &comm) {
  return Impl::send<SendMode>(space, sv, dest, tag, comm.as_raw());
}

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void recv(const ExecSpace &space, RecvView &sv, int src, int tag, const Communicator<ExecSpace> &comm) {
  return Impl::recv(space, sv, src, tag, comm.as_raw());
}

}  // namespace KokkosComm
