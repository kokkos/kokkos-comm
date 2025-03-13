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
#include "concepts.hpp"
#include "point_to_point.hpp"
#include "collective.hpp"

// Communication spaces declarations
#if defined(KOKKOSCOMM_ENABLE_MPI)
#include "mpi/mpi_space.hpp"

#include "mpi/comm_mode.hpp"
#include "mpi/handle.hpp"
#include "mpi/req.hpp"

#include "mpi/irecv.hpp"
#include "mpi/isend.hpp"
#include "mpi/recv.hpp"
#include "mpi/send.hpp"

#include "mpi/allgather.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/reduce.hpp"

#include "mpi/barrier.hpp"
#else
#error at least one transport must be defined
#endif

namespace KokkosComm {}  // namespace KokkosComm
