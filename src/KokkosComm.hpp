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

#include "KokkosComm_collective.hpp"
#include "KokkosComm_version.hpp"
#include "KokkosComm_isend.hpp"
#include "KokkosComm_irecv.hpp"
#include "KokkosComm_recv.hpp"
#include "KokkosComm_send.hpp"
#include "KokkosComm_alltoall.hpp"
#include "KokkosComm_barrier.hpp"
#include "KokkosComm_concepts.hpp"
#include "KokkosComm_CommModes.hpp"

#include <Kokkos_Core.hpp>

namespace KokkosComm {

using Impl::alltoall;
using Impl::barrier;
using Impl::irecv;
using Impl::isend;
using Impl::recv;
using Impl::send;

}  // namespace KokkosComm
