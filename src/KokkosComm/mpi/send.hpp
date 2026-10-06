// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/error.hpp>

#include "comm_mode.hpp"

#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm::mpi {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, CommunicationMode SendMode>
KokkosComm::status_type send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm, SendMode) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::send");
  using T      = typename SendView::non_const_value_type;
  using Packer = typename Impl::PackTraits<SendView>::packer_type;

  auto mpi_send_fn = [dest, tag, comm](void *view, int cnt, MPI_Datatype dtype) -> KokkosComm::status_type {
    if constexpr (std::is_same_v<SendMode, CommModeStandard>) {
      KC_MPI_CHECK(MPI_Send(view, cnt, dtype, dest, tag, comm), "KokkosComm::mpi::send");
    } else if constexpr (std::is_same_v<SendMode, CommModeReady>) {
      KC_MPI_CHECK(MPI_Rsend(view, cnt, dtype, dest, tag, comm), "KokkosComm::mpi::send");
    } else if constexpr (std::is_same_v<SendMode, CommModeSynchronous>) {
      KC_MPI_CHECK(MPI_Ssend(view, cnt, dtype, dest, tag, comm), "KokkosComm::mpi::send");
    } else {
      static_assert(std::is_void_v<SendMode>, "KokkosComm::mpi::send: unexpected communication mode");
      return {};
    }
  };

  if (is_contiguous(sv)) {
    space.fence("fence before send");
    if (auto r = mpi_send_fn(data_handle(sv), span(sv), datatype<MpiSpace, T>()); !r.has_value())
      return tl::unexpected(r.error());
  } else {
    auto args = Packer::pack(space, "pkd_sv", sv);
    space.fence("fence before send");
    if (auto r = mpi_send_fn(data_handle(args.view), args.count, args.datatype); !r.has_value())
      return tl::unexpected(r.error());
  }

  Kokkos::Tools::popRegion();
  return {};
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
KokkosComm::status_type send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  return send(space, sv, dest, tag, comm, DefaultCommMode{});
}

/// NOTE: This overload has the side effect of fencing on the default execution space.
template <KokkosView SendView>
KokkosComm::status_type send(const SendView &sv, int dest, int tag, MPI_Comm comm) {
  return send(Kokkos::DefaultExecutionSpace(), sv, dest, tag, comm, DefaultCommMode{});
}

}  // namespace KokkosComm::mpi
