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

#include "KokkosComm_fwd.hpp"

// transport declarations
// TODO: could probably be moved to a per-transport file to be included
#if defined(KOKKOSCOMM_TRANSPORT_MPI)
#include "mpi/KokkosComm_mpi.hpp"
#include "mpi/KokkosComm_mpi_handle.hpp"
#include "mpi/KokkosComm_mpi_plan.hpp"
#include "mpi/KokkosComm_mpi_isend.hpp"
#include "mpi/KokkosComm_mpi_irecv.hpp"
#else
#error at least one transport must be defined
#endif

#include "KokkosComm_point_to_point.hpp"
#include "KokkosComm_collective.hpp"
#include "KokkosComm_version.hpp"
#include "KokkosComm_isend.hpp"
#include "KokkosComm_irecv.hpp"
#include "KokkosComm_recv.hpp"
#include "KokkosComm_send.hpp"
#include "KokkosComm_alltoall.hpp"
#include "KokkosComm_barrier.hpp"
#include "KokkosComm_concepts.hpp"
#include "KokkosComm_comm_mode.hpp"

namespace KokkosComm {

using Impl::alltoall;
using Impl::barrier;
using Impl::recv;
using Impl::send;

}  // namespace KokkosComm
