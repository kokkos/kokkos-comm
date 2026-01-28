// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "comm_mode.hpp"

#include <KokkosComm/impl/host_staging.hpp>
#include "impl/pack_traits.hpp"

namespace KokkosComm::mpi {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, CommunicationMode SendMode>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm, SendMode) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::send");
  using T      = typename SendView::non_const_value_type;
  using Packer = typename KokkosComm::PackTraits<SendView>::packer_type;

  auto mpi_send_fn = [dest, tag, comm](void *view, int cnt, MPI_Datatype dtype) {
    if constexpr (std::is_same_v<SendMode, CommModeStandard>) {
      MPI_Send(view, cnt, dtype, dest, tag, comm);
    } else if constexpr (std::is_same_v<SendMode, CommModeReady>) {
      MPI_Rsend(view, cnt, dtype, dest, tag, comm);
    } else if constexpr (std::is_same_v<SendMode, CommModeSynchronous>) {
      MPI_Ssend(view, cnt, dtype, dest, tag, comm);
    } else {
      static_assert(std::is_void_v<SendMode>, "KokkosComm::mpi::send: unexpected communication mode");
    }
  };

#if defined(KOKKOSCOMM_ENABLE_GPU_AWARE_MPI)
  if (is_contiguous(sv)) {
    space.fence("fence before GPU-aware `MPI_Send`");
    mpi_send_fn(data_handle(sv), span(sv), datatype<MpiSpace, T>());
  } else {
    auto args = Packer::pack(space, sv);
    space.fence("fence packing before GPU-aware `MPI_Send`");
    mpi_send_fn(data_handle(args.view), args.count, args.datatype);
  }
#else
  auto host_sv = KokkosComm::Impl::stage_for(sv);
  space.fence("fence host staging before `MPI_Send`");
  if (is_contiguous(host_sv)) {
    mpi_send_fn(data_handle(host_sv), span(host_sv), datatype<MpiSpace, T>());
  } else {
    auto args = Packer::pack(space, host_sv);
    space.fence("fence packing before `MPI_Send`");
    mpi_send_fn(data_handle(args.view), args.count, args.datatype);
  }
#endif

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  send(space, sv, dest, tag, comm, DefaultCommMode{});
}

/// NOTE: This overload has the side effect of fencing on the default execution space.
template <KokkosView SendView>
void send(const SendView &sv, int dest, int tag, MPI_Comm comm) {
  send(Kokkos::DefaultExecutionSpace(), sv, dest, tag, comm, DefaultCommMode{});
}

}  // namespace KokkosComm::mpi
