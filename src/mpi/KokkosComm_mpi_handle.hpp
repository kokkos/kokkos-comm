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
  using transport_type  = Mpi;
  using size_type       = int;

  explicit Handle(const execution_space &space, MPI_Comm comm) : space_(space), comm_(comm) {}
  explicit Handle(MPI_Comm comm) : Handle(Kokkos::DefaultExecutionSpace{}, comm) {}
  Handle() : Handle(Kokkos::DefaultExecutionSpace{}, MPI_COMM_WORLD) {}

  MPI_Comm &mpi_comm() { return comm_; }
  const execution_space &space() const { return space_; }

  size_type size() {
    size_type ret;
    MPI_Comm_size(comm_, &ret);
    return ret;
  }

  size_type rank() {
    size_type ret;
    MPI_Comm_rank(comm_, &ret);
    return ret;
  }

 private:
  execution_space space_;
  MPI_Comm comm_;
};

}  // namespace KokkosComm