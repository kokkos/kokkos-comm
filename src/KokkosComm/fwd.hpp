// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <KokkosComm/config.hpp>
#include "concepts.hpp"

namespace KokkosComm {

#if defined(KOKKOSCOMM_ENABLE_MPI)
struct Mpi;
using DefaultCommunicationSpace  = Mpi;
using FallbackCommunicationSpace = Mpi;
#else
#error at least one transport must be defined
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
template <KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
struct Barrier;

}  // namespace Impl
}  // namespace KokkosComm
