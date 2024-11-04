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

#include <KokkosComm/fwd.hpp>

// transport declarations
// TODO: could probably be moved to a per-transport file to be included
#if defined(KOKKOSCOMM_ENABLE_MPI)
#include <KokkosComm/mpi/mpi.hpp>
#include <KokkosComm/mpi/send.hpp>
#include <KokkosComm/mpi/allgather.hpp>
#include <KokkosComm/mpi/alltoall.hpp>
#include <KokkosComm/mpi/barrier.hpp>
#include <KokkosComm/mpi/handle.hpp>
#include <KokkosComm/mpi/irecv.hpp>
#include <KokkosComm/mpi/isend.hpp>
#include <KokkosComm/mpi/recv.hpp>
#include <KokkosComm/mpi/reduce.hpp>
#else
#error at least one transport must be defined
#endif

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/point_to_point.hpp>
#include <KokkosComm/collective.hpp>

#include <Kokkos_Core.hpp>

namespace KokkosComm {}  // namespace KokkosComm
