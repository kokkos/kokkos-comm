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

#include
#include <KokkosComm/fwd.hpp>
#include <KokkosComm/nccl/nccl.hpp>

namespace KokkosComm::Experimental {

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
class Handle<ExecSpace, Nccl> {
 public:
  using execution_space = ExecSpace;
  using transport_type  = Nccl;
  using size_type       = int;

  explicit Handle(const execution_space &space, ncclComm_t comm) : space_(space), comm_(comm) {}
  explicit Handle(ncclComm_t comm) : Handle(execution_space{}, comm) {}

  // NOTE: Do we want to allow users creating a NCCL Handle without providing the communicator?
  // This would require us initializing it manually, which is a lot more work than for initializing MPI.
  //
  // Commenting it out for now.
  // Handle() : Handle(Kokkos::DefaultExecutionSpace{}, ) {}

  auto get_inner() -> ncclComm_t & { return comm_; }
  auto space() const -> const execution_space & { return space_; }

  auto size() -> size_type {
    size_type ret;
    ncclCommCount(comm_, &ret);
    return ret;
  }

  auto rank() -> size_type {
    size_type ret;
    ncclCommUserRank(comm_, &ret);
    return ret;
  }

 private:
  execution_space space_;
  ncclComm_t comm_;
};

}  // namespace KokkosComm::Experimental
