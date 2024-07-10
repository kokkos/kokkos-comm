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

#include "KokkosComm_fwd.hpp"

#include "KokkosComm_mpi_req.hpp"

namespace KokkosComm {

/*
- init_fence
- allocations
- pre_copies
- pre_comm_fence
- comm

- wait
- post-wait
*/
template <KokkosExecutionSpace ExecSpace>
class Handle<ExecSpace, Mpi> {
 public:
  using execution_space = ExecSpace;

  Handle(const execution_space &space, MPI_Comm comm) : space_(space), comm_(comm), preCommFence_(false) {}

  MPI_Comm &mpi_comm() { return comm_; }
  const execution_space &space() const { return space_; }

  void impl_add_pre_comm_fence() { preCommFence_ = true; }

  void impl_add_alloc(std::function<void()> f) { allocs_.push_back(f); }

  void impl_add_pre_copy(std::function<void()> f) { preCopies_.push_back(f); }

  void impl_add_comm(std::function<void()> f) { comms_.push_back(f); }

  void impl_track_req(const Req<Mpi> &req) { reqs_.push_back(req); }

  void impl_run() {
    for (const auto &f : allocs_) f();
    for (const auto &f : preCopies_) f();
    if (preCommFence_) {
      space_.fence("pre-comm fence");
    }
    for (const auto &f : comms_) f();

    allocs_.clear();
    preCopies_.clear();
    comms_.clear();
  }

  std::vector<Req<Mpi>> &impl_reqs() { return reqs_; }

 private:
  execution_space space_;
  MPI_Comm comm_;

  // phase variables
  bool preCommFence_;
  std::vector<std::function<void()>> allocs_;
  std::vector<std::function<void()>> preCopies_;
  std::vector<std::function<void()>> comms_;

  // wait variables
  std::vector<Req<Mpi>> reqs_;
  std::vector<std::function<void()>> postWaits_;
};

}  // namespace KokkosComm
