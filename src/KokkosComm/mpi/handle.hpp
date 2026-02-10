// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <KokkosComm/fwd.hpp>
#include <KokkosComm/concepts.hpp>
#include "mpi_space.hpp"

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
class Handle<ExecSpace, MpiSpace> {
 public:
  using execution_space = ExecSpace;
  using transport_type  = MpiSpace;
  using size_type       = int;

  explicit Handle(const execution_space& space, MPI_Comm comm) : space_(space), comm_(comm) {}
  explicit Handle(MPI_Comm comm) : Handle(Kokkos::DefaultExecutionSpace{}, comm) {}
  Handle() : Handle(Kokkos::DefaultExecutionSpace{}, MPI_COMM_WORLD) {}

  MPI_Comm& mpi_comm() { return comm_; }
  const execution_space& space() const { return space_; }

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
