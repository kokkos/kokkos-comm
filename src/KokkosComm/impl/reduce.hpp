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

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <KokkosComm/nccl/handle.hpp>
#include <KokkosComm/nccl/reduce.hpp>
#endif

#include <Kokkos_Core.hpp>

namespace KokkosComm::Experimental::Impl {

#if defined(KOKKOSCOMM_ENABLE_NCCL)
template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::Cuda, CommunicationSpace CommSpace = Nccl>
struct Reduce {
  static auto execute(Handle<ExecSpace, CommSpace> &h, const SendView sv, RecvView, int root) -> Req<CommSpace> {
    KokkosComm::Experimental::nccl::Impl::reduce(
        h.space(), sv, rv, KokkosComm::Experimental::nccl::Impl::reduction_op_v<RedOp>, root, h.rank(), h.comm());
    return Req<CommSpace>(h.space().cuda_stream());
  }
};
#endif

}  // namespace KokkosComm::Experimental::Impl
