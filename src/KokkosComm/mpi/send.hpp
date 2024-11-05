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

#include <KokkosComm/mpi/commmode.hpp>
#include <KokkosComm/mpi/impl/pack_traits.hpp>
#include <KokkosComm/mpi/impl/include_mpi.hpp>

#include <Kokkos_Core.hpp>

namespace KokkosComm::mpi {

template <KokkosView SendView, CommunicationMode SendMode>
void send(const SendView &sv, int dest, int tag, MPI_Comm comm, SendMode) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::send");
  using KCT = typename KokkosComm::Traits<SendView>;

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
    using SendScalar = typename SendView::non_const_value_type;
    MPI_Send(KokkosComm::data_handle(sv), KokkosComm::span(sv), KokkosComm::Impl::mpi_type_v<SendScalar>, dest, tag,
             comm);
  } else {
    throw std::runtime_error("only contiguous views supported for low-level send");
  }
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
    mpi_send_fn(sv.data(), sv.span(), KokkosComm::Impl::mpi_type_v<SendScalar>, dest, tag, comm);
  } else {
    auto args = Packer::pack(space, sv);
    space.fence();
    mpi_send_fn(args.view.data(), args.count, args.datatype, dest, tag, comm);
  }

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  send(space, sv, dest, tag, comm, DefaultCommMode{});
}

}  // namespace KokkosComm::mpi
