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

#include "KokkosComm_config.hpp"
#include "KokkosComm_collective.hpp"
#include "impl/KokkosComm_isend.hpp"
#include "impl/KokkosComm_irecv.hpp"
#include "impl/KokkosComm_recv.hpp"
#include "impl/KokkosComm_send.hpp"
#include "impl/KokkosComm_alltoall.hpp"
#include "impl/KokkosComm_barrier.hpp"
#include "impl/KokkosComm_concepts.hpp"
#include "KokkosComm_comm_modes.hpp"

#include <Kokkos_Core.hpp>

namespace KokkosComm {

using Impl::alltoall;
using Impl::barrier;
using Impl::irecv;
using Impl::isend;
using Impl::recv;
using Impl::send;

}  // namespace KokkosComm
