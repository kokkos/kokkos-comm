// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core_fwd.hpp>

#include "fwd.hpp"
#include "reduction_op.hpp"
#if defined(KOKKOSCOMM_ENABLE_MPI)
#include "mpi/mpi_space.hpp"
#include "mpi/handle.hpp"
#include "mpi/request.hpp"
#include "mpi/broadcast.hpp"
#include "mpi/allgather.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/allreduce.hpp"
#include "mpi/reduce.hpp"
#endif
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/nccl_space.hpp"
#include "nccl/handle.hpp"
#include "nccl/request.hpp"
#include "nccl/broadcast.hpp"
#include "nccl/allgather.hpp"
#include "nccl/alltoall.hpp"
#include "nccl/allreduce.hpp"
#include "nccl/reduce.hpp"
#endif

namespace KokkosComm::Experimental {

/// Copy the `v` view from the `root` rank to all ranks' `v` view.
template <KokkosView View, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto broadcast(Handle<ExecSpace, CommSpace>& h, View v, int root) -> Request<CommSpace> {
  return Impl::Broadcast<View, ExecSpace, CommSpace>::execute(h, v, root);
}

/// Copy the `sv` view from each rank to the `rv` view, receiving data from rank `i` at offset
/// `i * KokkosComm::span(sv)`.
///
/// Note: this assumes the span of the `rv` view to be `h.size() * KokkosComm::span(sv)`.
template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto allgather(Handle<ExecSpace, CommSpace>& h, const SendView sv, RecvView rv) -> Request<CommSpace> {
  return Impl::AllGather<SendView, RecvView, ExecSpace, CommSpace>::execute(h, sv, rv);
}

/// Send `count` elements from the `sv` view, and receive `count` elements from all other ranks to the `rv` view.
///
/// Data to send to destination rank `i` is taken from the `sv` view at offset `i * count`.
/// Data to receive from source rank `j` is placed into the `rv` view at offset `j * count`.
///
/// Note: this assumes the span of both `sv` and `rv` views to be `h.size * count`.
template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
auto alltoall(Handle<ExecSpace, CommSpace>& h, const SendView sv, RecvView rv, int count) -> Request<CommSpace> {
  return Impl::AllToAll<SendView, RecvView, ExecSpace, CommSpace>::execute(h, sv, rv, count);
}

/// Reduce the `sv` view using the `RedOp` operation and copy the result to all ranks' `rv` view.
template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
auto allreduce(Handle<ExecSpace, CommSpace>& h, const SendView sv, RecvView rv, RedOp) -> Request<CommSpace> {
  return Impl::AllReduce<SendView, RecvView, RedOp, ExecSpace, CommSpace>::execute(h, sv, rv);
}

/// Reduce the `sv` view using the `RedOp` operation and copy the result to the `root` rank's `rv` view.
///
/// The `rv` view is only used on the `root` rank and ignored for all other ranks.
template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
auto reduce(Handle<ExecSpace, CommSpace>& h, const SendView sv, RecvView rv, int root, RedOp) -> Request<CommSpace> {
  return Impl::Reduce<SendView, RecvView, RedOp, ExecSpace, CommSpace>::execute(h, sv, rv, root);
}

}  // namespace KokkosComm::Experimental
