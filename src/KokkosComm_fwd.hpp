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

#include "KokkosComm_concepts.hpp"
#include "KokkosComm_config.hpp"

namespace KokkosComm {
#if defined(KOKKOSCOMM_TRANSPORT_MPI)
class Mpi;
using DefaultTransport  = Mpi;
using FallbackTransport = Mpi;
#else
#error at least one transport must be defined
#endif

template <KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, Transport TRANSPORT = DefaultTransport>
class Handle;

template <Dispatch DISPATCH, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
class Plan;

namespace Impl {

template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
struct Irecv;
template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
struct Isend;

}  // namespace Impl

}  // namespace KokkosComm
