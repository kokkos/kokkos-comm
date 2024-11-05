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

#include "fwd.hpp"

// transport declarations
// TODO: could probably be moved to a per-transport file to be included
#if defined(KOKKOSCOMM_ENABLE_MPI)
#include "mpi/mpi.hpp"
#include "mpi/send.hpp"
#include "mpi/allgather.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/barrier.hpp"
#include "mpi/handle.hpp"
#include "mpi/irecv.hpp"
#include "mpi/isend.hpp"
#include "mpi/recv.hpp"
#include "mpi/reduce.hpp"
#include "mpi/impl/wait.hpp"
#else
#error at least one transport must be defined
#endif

#include "concepts.hpp"
#include "point_to_point.hpp"
#include "collective.hpp"
#include "wait.hpp"

#include <Kokkos_Core.hpp>

namespace KokkosComm {}  // namespace KokkosComm
