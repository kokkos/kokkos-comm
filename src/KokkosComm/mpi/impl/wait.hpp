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

#include <KokkosComm/mpi/req.hpp>

namespace KokkosComm::Impl {

/* Enqueue a communication completion*/
template <KokkosExecutionSpace ExecSpace>
struct Wait<ExecSpace, Mpi> {
  Wait(const ExecSpace &space, Req<Mpi> req) {
    // ensure that the execution space has completed all work before completing the communication
    space.fence();
    MPI_Wait(&req.mpi_request(), MPI_STATUS_IGNORE);
    for (auto &f : req.record_->postWaits_) {
      f();
    }
    req.record_->postWaits_.clear();
  }
};

template <KokkosExecutionSpace ExecSpace>
struct WaitAll<ExecSpace, Mpi> {
  WaitAll(const ExecSpace &space, std::vector<Req<Mpi>> &reqs) {
    // ensure that the execution space has completed all work before completing the communication
    space.fence();
    for (Req<Mpi> &req : reqs) {
      MPI_Wait(&req.mpi_request(), MPI_STATUS_IGNORE);
      for (auto &f : req.record_->postWaits_) {
        f();
      }
      req.record_->postWaits_.clear();
    }
  }
};

/* Returns the index of the request that completed */
template <KokkosExecutionSpace ExecSpace>
struct WaitAny<ExecSpace, Mpi> {
  static int execute(const ExecSpace &space, std::vector<Req<Mpi>> &reqs) {
    if (reqs.empty()) {
      return -1;
    }

    // ensure that the execution space has completed all work before completing the communication
    space.fence();
    while (true) {  // wait until something is done
      for (size_t i = 0; i < reqs.size(); ++i) {
        int completed;
        Req<Mpi> &req = reqs[i];
        MPI_Test(&(req.mpi_request()), &completed, MPI_STATUS_IGNORE);
        if (completed) {
          for (auto &f : req.record_->postWaits_) {
            f();
          }
          req.record_->postWaits_.clear();
          return i;
        }
      }
    }
  }
};

}  // namespace KokkosComm::Impl