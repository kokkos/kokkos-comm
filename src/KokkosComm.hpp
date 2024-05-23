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

#include "KokkosComm_concepts.hpp"
#include "KokkosComm_mpi.hpp"
#include "KokkosComm_comm_mode.hpp"
#include "KokkosComm_version.hpp"

#include "KokkosComm_send.hpp"
#include "KokkosComm_recv.hpp"
#include "KokkosComm_sendrecv.hpp"
#include "KokkosComm_reduce.hpp"
#include "KokkosComm_allreduce.hpp"
#include "KokkosComm_alltoall.hpp"

#include "KokkosComm_isend.hpp"
#include "KokkosComm_irecv.hpp"
#include "KokkosComm_isendrecv.hpp"
#include "KokkosComm_ireduce.hpp"
#include "KokkosComm_iallreduce.hpp"

#include "KokkosComm_barrier.hpp"

namespace KokkosComm {
using Impl::Communicator;
using Impl::CommWorld;
using Impl::CommSelf;

using Impl::Reducer;
using Impl::Sum;
using Impl::Min;
using Impl::Max;
using Impl::LogicalOr;

using Impl::Request;

using Impl::send;
using Impl::recv;
using Impl::sendrecv;
using Impl::reduce;
using Impl::allreduce;

using Impl::isend;
using Impl::irecv;
using Impl::isendrecv;
using Impl::ireduce;
using Impl::iallreduce;

inline int size(Communicator comm) { return comm.size(); }
inline int rank(Communicator comm) { return comm.rank(); }
inline void barrier(Communicator comm) { comm.barrier(); }
inline Request ibarrier(Communicator comm) { return comm.ibarrier(); }
}  // namespace KokkosComm
