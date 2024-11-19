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

#include <KokkosComm/fwd.hpp>

#include <nccl.h>

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
  using execution_space     = ExecSpace;
  using communication_space = Nccl;
  using communicator_type   = ncclComm_t;
  using datatype_type       = ncclDataType_t;
  using reduction_op_type   = ncclRedOp_t;
  using rank_type           = int;

  explicit Handle(const execution_space &space, communicator_type comm) : space_(space), comm_(comm) {}
  explicit Handle(communicator_type comm) : Handle(execution_space{}, comm) {}

  // NOTE: Do we want to allow users creating a NCCL Handle without providing the communicator?
  // This would require us initializing it manually, which is a lot more work than for initializing MPI.
  //
  // Commenting it out for now.
  // Handle() : Handle(Kokkos::DefaultExecutionSpace{}, ) {}

  auto get_inner() -> ncclComm_t & { return comm_; }
  auto space() const -> const execution_space & { return space_; }

  auto size() -> rank_type {
    rank_type ret;
    ncclCommCount(comm_, &ret);
    return ret;
  }

  auto rank() -> rank_type {
    rank_type ret;
    ncclCommUserRank(comm_, &ret);
    return ret;
  }

 private:
  execution_space space_;
  communicator_type comm_;
};

}  // namespace KokkosComm::Experimental
