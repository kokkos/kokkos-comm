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

#include "KokkosComm_fwd.hpp"

namespace KokkosComm {

template <Dispatch TDispatch, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
std::vector<Req<TRANSPORT>> plan(Handle<ExecSpace, TRANSPORT> &handle, TDispatch d) {
  return Impl::Plan<TDispatch, ExecSpace, TRANSPORT>(handle, d).reqs;
}

template <Dispatch TDispatch, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          Transport TRANSPORT = DefaultTransport>
std::vector<Req<TRANSPORT>> plan(const ExecSpace &space, MPI_Comm comm, TDispatch d) {
  Handle<ExecSpace, TRANSPORT> handle(space, comm);
  auto ret = plan<TDispatch, ExecSpace, TRANSPORT>(handle, d);
  return ret;
}

}  // namespace KokkosComm
