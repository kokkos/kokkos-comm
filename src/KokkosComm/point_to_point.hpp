// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core_fwd.hpp>

#include "fwd.hpp"
#if defined(KOKKOSCOMM_ENABLE_MPI)
#include "mpi/mpi_space.hpp"
#include "mpi/communicator.hpp"
#include "mpi/request.hpp"
#include "mpi/isend.hpp"
#include "mpi/irecv.hpp"
#endif
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/nccl_space.hpp"
#include "nccl/communicator.hpp"
#include "nccl/request.hpp"
#include "nccl/send.hpp"
#include "nccl/recv.hpp"
#endif

namespace KokkosComm {

/// Send w/ explicit handle
template <
    KokkosView SendView,
    KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
    CommunicationSpace CommSpace   = DefaultCommunicationSpace>
auto send(Communicator<CommSpace, ExecSpace>& h, const SendView& sv, int peer) -> Request<CommSpace> {
  return Impl::Send<SendView, ExecSpace, CommSpace>::execute(h, sv, peer);
}

/// Receive w/ explicit handle
template <
    MutKokkosView RecvView,
    KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
    CommunicationSpace CommSpace   = DefaultCommunicationSpace>
auto recv(Communicator<CommSpace, ExecSpace>& h, RecvView& rv, int peer) -> Request<CommSpace> {
  return Impl::Recv<RecvView, ExecSpace, CommSpace>::execute(h, rv, peer);
}

}  // namespace KokkosComm
