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

#include "KokkosComm/traits.hpp"
#include "impl/pack_traits.hpp"
#include "impl/include_mpi.hpp"
#include "impl/types.hpp"

namespace KokkosComm::mpi {

template <KokkosView SendView, KokkosView RecvView>
void allgather(const SendView &sv, const RecvView &rv, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Mpi::allgather");

  using SendScalar = typename SendView::value_type;
  using RecvScalar = typename RecvView::value_type;

  static_assert(KokkosComm::rank<SendView>() <= 1, "allgather for SendView::rank > 1 not supported");
  static_assert(KokkosComm::rank<RecvView>() <= 1, "allgather for RecvView::rank > 1 not supported");

  if (!KokkosComm::is_contiguous(sv)) {
    throw std::runtime_error("low-level allgather requires contiguous send view");
  }
  if (!KokkosComm::is_contiguous(rv)) {
    throw std::runtime_error("low-level allgather requires contiguous recv view");
  }
  const int count = KokkosComm::span(sv);  // all ranks send/recv same count
  MPI_Allgather(KokkosComm::data_handle(sv), count, KokkosComm::Impl::mpi_type_v<SendScalar>,
                KokkosComm::data_handle(rv), count, KokkosComm::Impl::mpi_type_v<RecvScalar>, comm);

  Kokkos::Tools::popRegion();
}

// in-place allgather
template <KokkosView RecvView>
void allgather(const RecvView &rv, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Mpi::allgather");

  using RT         = KokkosComm::Traits<RecvView>;
  using RecvScalar = typename RecvView::value_type;

  static_assert(RT::rank() <= 1, "allgather for RecvView::rank > 1 not supported");

  if (!RT::is_contiguous(rv)) {
    throw std::runtime_error("low-level allgather requires contiguous recv view");
  }
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, RT::data_handle(rv), RT::span(rv),
                KokkosComm::Impl::mpi_type_v<RecvScalar>, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void allgather(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Mpi::allgather");
  using SPT = KokkosComm::PackTraits<SendView>;
  using RPT = KokkosComm::PackTraits<RecvView>;

  if (!KokkosComm::is_contiguous(sv) || !KokkosComm::is_contiguous(rv)) {
    throw std::runtime_error("allgather for non-contiguous views not implemented");
  } else {
    space.fence("fence before allgather");  // work in space may have been used to produce send view data
    allgather(sv, rv, comm);
  }

  Kokkos::Tools::popRegion();
}
}  // namespace KokkosComm::mpi
