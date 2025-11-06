// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <KokkosComm/config.hpp>
#include "concepts.hpp"
#include "reduction_op.hpp"

namespace KokkosComm {

// NCCL backend also implicitly declares MPI
#if defined(KOKKOSCOMM_ENABLE_NCCL)
namespace Experimental {
class Nccl;
}  // namespace Experimental
class Mpi;
using DefaultCommunicationSpace  = Experimental::Nccl;
using FallbackCommunicationSpace = Mpi;
#elif defined(KOKKOSCOMM_ENABLE_MPI)
class Mpi;
using DefaultCommunicationSpace  = Mpi;
using FallbackCommunicationSpace = Mpi;
#else
#error at least one communication space must be enabled
#endif

template <CommunicationSpace CommSpace = DefaultCommunicationSpace>
class Req;

template <KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
class Handle;

namespace Impl {

template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
struct Recv;

template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
struct Send;

}  // namespace Impl

// Collectives are currently experimental functions
namespace Experimental::Impl {

template <KokkosView View, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
struct Broadcast;

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
struct AllGather;

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace = DefaultCommunicationSpace>
struct AllToAll;

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct AllReduce;

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp,
          KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct Reduce;

}  // namespace Experimental::Impl

}  // namespace KokkosComm
