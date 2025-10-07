// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include "comm_mode.hpp"

#include "impl/pack_traits.hpp"
#include "impl/types.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm::mpi {

template <KokkosView SendView, CommunicationMode SendMode>
void send(const SendView &sv, int dest, int tag, MPI_Comm comm, SendMode) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::send");

  auto mpi_send_fn = [](void *mpi_view, int mpi_count, MPI_Datatype mpi_datatype, int mpi_dest, int mpi_tag,
                        MPI_Comm mpi_comm) {
    if constexpr (std::is_same_v<SendMode, CommModeStandard>) {
      MPI_Send(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm);
    } else if constexpr (std::is_same_v<SendMode, CommModeReady>) {
      MPI_Rsend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm);
    } else if constexpr (std::is_same_v<SendMode, CommModeSynchronous>) {
      MPI_Ssend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm);
    } else {
      static_assert(std::is_void_v<SendMode>, "unexpected communication mode");
    }
  };

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(sv), "only contiguous views supported for low-level send");

  using SendScalar = typename SendView::non_const_value_type;
  MPI_Send(KokkosComm::data_handle(sv), KokkosComm::span(sv), KokkosComm::Impl::mpi_type_v<SendScalar>, dest, tag,
           comm);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, CommunicationMode SendMode>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm, SendMode) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::send");

  using Packer = typename KokkosComm::PackTraits<SendView>::packer_type;

  auto mpi_send_fn = [](void *mpi_view, int mpi_count, MPI_Datatype mpi_datatype, int mpi_dest, int mpi_tag,
                        MPI_Comm mpi_comm) {
    if constexpr (std::is_same_v<SendMode, CommModeStandard>) {
      MPI_Send(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm);
    } else if constexpr (std::is_same_v<SendMode, CommModeReady>) {
      MPI_Rsend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm);
    } else if constexpr (std::is_same_v<SendMode, CommModeSynchronous>) {
      MPI_Ssend(mpi_view, mpi_count, mpi_datatype, mpi_dest, mpi_tag, mpi_comm);
    } else {
      static_assert(std::is_void_v<SendMode>, "unexpected communication mode");
    }
  };

  if (KokkosComm::is_contiguous(sv)) {
    using SendScalar = typename SendView::value_type;
    space.fence("fence before MPI_Send");
    mpi_send_fn(KokkosComm::data_handle(sv), KokkosComm::span(sv), KokkosComm::Impl::mpi_type_v<SendScalar>, dest, tag,
                comm);
  } else {
    auto args = Packer::pack(space, sv);
    space.fence("fence before MPI_Send");
    mpi_send_fn(KokkosComm::data_handle(args.view), args.count, args.datatype, dest, tag, comm);
  }

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  send(space, sv, dest, tag, comm, DefaultCommMode{});
}

}  // namespace KokkosComm::mpi
