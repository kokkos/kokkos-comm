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

#include "fwd.hpp"
#include "concepts.hpp"

namespace KokkosComm {

// FIXME: reverse order of these template params for automatic deduction
template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
void wait(const ExecSpace &space, Req<CommSpace> req) {
  Impl::Wait<ExecSpace, CommSpace>(space, req);
}

template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
void wait_all(const ExecSpace &space, std::vector<Req<Mpi>> &reqs) {
  Impl::WaitAll<ExecSpace, CommSpace>(space, reqs);
}

template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
int wait_any(const ExecSpace &space, std::vector<Req<Mpi>> &reqs) {
  return Impl::WaitAny<ExecSpace, CommSpace>::execute(space, reqs);
}

template <CommunicationSpace CommSpace>
inline void wait(Req<CommSpace> req) {
  return wait<Kokkos::DefaultExecutionSpace, CommSpace>(Kokkos::DefaultExecutionSpace{}, req);
}
template <CommunicationSpace CommSpace>
inline void wait_all(std::vector<Req<CommSpace>> &reqs) {
  wait_all<Kokkos::DefaultExecutionSpace, CommSpace>(Kokkos::DefaultExecutionSpace{}, reqs);
}
template <CommunicationSpace CommSpace>
inline int wait_any(std::vector<Req<CommSpace>> &reqs) {
  return wait_any<Kokkos::DefaultExecutionSpace, CommSpace>(Kokkos::DefaultExecutionSpace{}, reqs);
}

}  // namespace KokkosComm
