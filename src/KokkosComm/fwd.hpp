// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <KokkosComm/config.hpp>
#include "concepts.hpp"
#include "datatype.hpp"
#include "reduction_op.hpp"

namespace KokkosComm {

#if defined(KOKKOSCOMM_ENABLE_NCCL)
namespace Experimental {
struct NcclSpace;
}
using DefaultCommunicationSpace = Experimental::NcclSpace;
#endif

#if defined(KOKKOSCOMM_ENABLE_MPI)
struct MpiSpace;
#if !defined(KOKKOSCOMM_ENABLE_NCCL)
using DefaultCommunicationSpace = MpiSpace;
#endif
#endif

#if !defined(KOKKOSCOMM_ENABLE_MPI) && !defined(KOKKOSCOMM_ENABLE_NCCL)
static_assert(false, "KokkosComm: at least one communication space must be defined");
#endif

/// @brief Template class for communicator wrappers.
template <
    CommunicationSpace Comm   = DefaultCommunicationSpace,
    KokkosExecutionSpace Exec = Kokkos::DefaultExecutionSpace>
class Communicator;

/// @brief Template class for request wrappers.
template <CommunicationSpace CommSpace = DefaultCommunicationSpace>
class Request;

namespace Impl {

template <
    KokkosView RecvView,
    KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
    CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct Recv;

template <
    KokkosView SendView,
    KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
    CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct Send;

}  // namespace Impl

// Collectives are currently experimental functions
namespace Experimental::Impl {

template <
    KokkosView View,
    KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
    CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct Broadcast;

template <
    KokkosView SendView,
    KokkosView RecvView,
    KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
    CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct AllGather;

template <
    KokkosView SendView,
    KokkosView RecvView,
    KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
    CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct AllToAll;

template <
    KokkosView SendView,
    KokkosView RecvView,
    ReductionOperator RedOp,
    KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
    CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct AllReduce;

template <
    KokkosView SendView,
    KokkosView RecvView,
    ReductionOperator RedOp,
    KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
    CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct Reduce;

}  // namespace Experimental::Impl

}  // namespace KokkosComm
