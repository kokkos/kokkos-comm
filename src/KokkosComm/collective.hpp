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
#include <KokkosComm/concepts.hpp>
#include <KokkosComm/nccl/reduce.hpp>

#include <Kokkos_Core.hpp>

#include <utility>
#include "KokkosComm/nccl/allgather.hpp"
#include "KokkosComm/nccl/nccl.hpp"

namespace KokkosComm {

template <KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
void barrier(Handle<ExecSpace, CommSpace>&& h) {
  Impl::Barrier<ExecSpace, CommSpace>{std::forward<Handle<ExecSpace, CommSpace>>(h)};
}

namespace Experimental {

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::Cuda,
          CommunicationSpace CommSpace = KokkosComm::Experimental::Nccl, KokkosComm::ReductionOp RedOp>
auto reduce(const Handle<ExecSpace, CommSpace>& h, const SendView& sv, RecvView& rv, int root) -> Req<Nccl> {
  nccl::Impl::reduce(h.space(), sv, rv, nccl::Impl::reduction_op_v<RedOp>, root, h.rank(), h.get_inner());
  return Req<Nccl>(h.space().cuda_stream());
}

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::Cuda,
          CommunicationSpace CommSpace = KokkosComm::Experimental::Nccl>
auto allgather(const Handle<ExecSpace, CommSpace>& h, const SendView& sv, RecvView& rv) -> Req<Nccl> {
  nccl::Impl::allgather(h.space(), sv, rv, h.get_inner());
  return Req<Nccl>(h.space().cuda_stream());
}

}  // namespace Experimental

}  // namespace KokkosComm
