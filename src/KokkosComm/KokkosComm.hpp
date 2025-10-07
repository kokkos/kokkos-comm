// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include "fwd.hpp"
#include "concepts.hpp"
#include "point_to_point.hpp"
#include "collective.hpp"

// Communication spaces declarations
#if defined(KOKKOSCOMM_ENABLE_MPI)
#include "mpi/mpi_space.hpp"

#include "mpi/channel.hpp"
#include "mpi/comm_mode.hpp"
#include "mpi/handle.hpp"
#include "mpi/req.hpp"

#include "mpi/irecv.hpp"
#include "mpi/isend.hpp"
#include "mpi/recv.hpp"
#include "mpi/send.hpp"

#include "mpi/broadcast.hpp"
#include "mpi/allgather.hpp"
#include "mpi/allreduce.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/reduce.hpp"
#include "mpi/scan.hpp"

#include "mpi/barrier.hpp"
#else
#error at least one transport must be defined
#endif

namespace KokkosComm {}  // namespace KokkosComm
