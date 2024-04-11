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

#include <Kokkos_Core.hpp>

#include "KokkosComm_pack_traits.hpp"

// impl
#include "KokkosComm_include_mpi.hpp"

namespace KokkosComm::Impl {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag,
          MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::send");

  using Packer = typename KokkosComm::PackTraits<SendView>::packer_type;

  if (KokkosComm::PackTraits<SendView>::needs_pack(sv)) {
    auto args = Packer::pack(space, sv);
    space.fence();
    MPI_Send(args.view.data(), args.count, args.datatype, dest, tag, comm);
  } else {
    using SendScalar = typename SendView::value_type;
    MPI_Send(sv.data(), sv.span(), mpi_type_v<SendScalar>, dest, tag, comm);
  }

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
void rsend(const ExecSpace &space, const SendView &sv, int dest, int tag,
           MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::rsend");

  using Packer = typename KokkosComm::PackTraits<SendView>::packer_type;

  if (KokkosComm::PackTraits<SendView>::needs_pack(sv)) {
    auto args = Packer::pack(space, sv);
    space.fence();
    MPI_Rsend(args.view.data(), args.count, args.datatype, dest, tag, comm);
  } else {
    using SendScalar = typename SendView::value_type;
    MPI_Rsend(sv.data(), sv.span(), mpi_type_v<SendScalar>, dest, tag, comm);
  }

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
void ssend(const ExecSpace &space, const SendView &sv, int dest, int tag,
           MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::ssend");

  using Packer = typename KokkosComm::PackTraits<SendView>::packer_type;

  if (KokkosComm::PackTraits<SendView>::needs_pack(sv)) {
    auto args = Packer::pack(space, sv);
    space.fence();
    MPI_Ssend(args.view.data(), args.count, args.datatype, dest, tag, comm);
  } else {
    using SendScalar = typename SendView::value_type;
    MPI_Ssend(sv.data(), sv.span(), mpi_type_v<SendScalar>, dest, tag, comm);
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::Impl
