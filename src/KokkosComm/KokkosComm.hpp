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
#include "mpi/request.hpp"

#include "mpi/irecv.hpp"
#include "mpi/isend.hpp"
#include "mpi/recv.hpp"
#include "mpi/send.hpp"

#include "mpi/broadcast.hpp"
#include "mpi/allgather.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/allreduce.hpp"
#include "mpi/reduce.hpp"
#include "mpi/scan.hpp"

#include "mpi/barrier.hpp"
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/nccl_space.hpp"

#include "nccl/handle.hpp"
#include "nccl/request.hpp"

#include "nccl/recv.hpp"
#include "nccl/send.hpp"

#include "nccl/broadcast.hpp"
#include "nccl/allgather.hpp"
#include "nccl/alltoall.hpp"
#include "nccl/allreduce.hpp"
#include "nccl/reduce.hpp"
#endif

#if !defined(KOKKOSCOMM_ENABLE_MPI) && !defined(KOKKOSCOMM_ENABLE_NCCL)
static_assert(false, "KokkosComm: at least one communication space must be defined");
#endif

namespace KokkosComm {}  // namespace KokkosComm
