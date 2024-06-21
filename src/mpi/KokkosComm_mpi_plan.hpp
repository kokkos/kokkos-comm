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

namespace KokkosComm::Impl {

template <Dispatch TDispatch, KokkosExecutionSpace ExecSpace>
struct Plan<TDispatch, ExecSpace, Mpi> {
  Plan(Handle<ExecSpace, Mpi> &handle, TDispatch d) {
    d(handle);
    handle.impl_run();
    reqs = handle.impl_reqs();
  }

  std::vector<Req<Mpi>> reqs;
};

}  // namespace KokkosComm::Impl
