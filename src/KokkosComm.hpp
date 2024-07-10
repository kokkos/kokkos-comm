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
#include "mpi/KokkosComm_mpi_send.hpp"
#include "mpi/KokkosComm_mpi_allgather.hpp"
#include "mpi/KokkosComm_mpi_alltoall.hpp"
#include "mpi/KokkosComm_mpi_barrier.hpp"
#include "mpi/KokkosComm_mpi_handle.hpp"
#include "mpi/KokkosComm_mpi_irecv.hpp"
#include "mpi/KokkosComm_mpi_isend.hpp"
#include "mpi/KokkosComm_mpi_recv.hpp"
#include "mpi/KokkosComm_mpi_reduce.hpp"
#else
#error at least one transport must be defined
#endif

#include "KokkosComm_version.hpp"

#include "KokkosComm_concepts.hpp"
#include "KokkosComm_point_to_point.hpp"
#include "KokkosComm_collective.hpp"

#include <Kokkos_Core.hpp>

namespace KokkosComm {}  // namespace KokkosComm
