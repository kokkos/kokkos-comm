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
#include "KokkosComm_traits.hpp"

// impl
#include "KokkosComm_include_mpi.hpp"
#include "KokkosComm_types.hpp"

namespace KokkosComm::Impl {
template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void allgather(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::allgather");

  using ST         = KokkosComm::Traits<SendView>;
  using RT         = KokkosComm::Traits<RecvView>;
  using SendScalar = typename SendView::value_type;
  using RecvScalar = typename RecvView::value_type;

  static_assert(ST::rank() <= 1, "allgather for SendView::rank > 1 not supported");
  static_assert(RT::rank() <= 1, "allgather for RecvView::rank > 1 not supported");

  if (KokkosComm::PackTraits<SendView>::needs_pack(sv) || KokkosComm::PackTraits<RecvView>::needs_pack(rv)) {
    throw std::runtime_error("allgather for non-contiguous views not implemented");
  } else {
    const int count = ST::span(sv);  // all ranks send/recv same count
    MPI_Allgather(ST::data_handle(sv), count, mpi_type_v<SendScalar>, RT::data_handle(rv), count,
                  mpi_type_v<RecvScalar>, comm);
  }

  Kokkos::Tools::popRegion();
}
}  // namespace KokkosComm::Impl
