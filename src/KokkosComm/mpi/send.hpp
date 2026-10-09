// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "comm_mode.hpp"

#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm::mpi {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, CommunicationMode SendMode>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm, SendMode) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::send");
  using T      = typename SendView::non_const_value_type;
  using Packer = typename Impl::PackTraits<SendView>::packer_type;

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

  if (is_contiguous(sv)) {
    space.fence("fence before send");
    mpi_send_fn(data_handle(sv), span(sv), datatype<MpiSpace, T>());
  } else {
    auto args = Packer::pack(space, "pkd_sv", sv);
    space.fence("fence before send");
    mpi_send_fn(data_handle(args.view), args.count, args.datatype);
  }

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
