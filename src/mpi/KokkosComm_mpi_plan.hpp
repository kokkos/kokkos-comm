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

namespace KokkosComm {

template <Dispatch DISPATCH, KokkosExecutionSpace ExecSpace>
class Plan<DISPATCH, ExecSpace, Mpi> {
 public:
  using execution_space = ExecSpace;
  using handle_type     = Handle<execution_space, Mpi>;
  Plan(const execution_space &space, MPI_Comm comm, DISPATCH d) : handle_(space, comm) {
    d(handle_);
    handle_.impl_run();
  }

  Plan(const execution_space &space, DISPATCH d) : Plan(space, MPI_COMM_WORLD, d) {}
  Plan(MPI_Comm comm, DISPATCH d) : Plan(Kokkos::DefaultExecutionSpace(), comm, d) {}
  Plan(DISPATCH d) : Plan(Kokkos::DefaultExecutionSpace(), MPI_COMM_WORLD, d) {}

  handle_type handle() const { return handle_; }

 private:
  handle_type handle_;
};

}  // namespace KokkosComm
