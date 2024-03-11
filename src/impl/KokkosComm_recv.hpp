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

#include <mpi.h>

#include <Kokkos_Core.hpp>

// impl
#include "KokkosComm_unpack.hpp"

/* FIXME: If RecvView is a Kokkos view, it can be a const ref
   same is true for an mdspan?
*/
namespace KokkosComm::Impl {
template <typename RecvView, typename ExecSpace>
void recv(const ExecSpace &space, RecvView &rv, int src, int tag,
          MPI_Comm comm) {

  using KCT = KokkosComm::Traits<RecvView>;

  if (KCT::needs_unpack(rv)) {
    using PackedScalar = typename KCT::packed_view_type::value_type;
    auto packed = allocate_packed_for(space, "packed", rv);
    space.fence();
    MPI_Recv(KCT::data_handle(packed), KCT::span(packed) * sizeof(PackedScalar),
             MPI_PACKED, src, tag, comm, MPI_STATUS_IGNORE);
    unpack(space, rv, packed);
    space.fence();
  } else {
    using RecvScalar = typename RecvView::value_type;
    MPI_Recv(KCT::data_handle(rv), KCT::span(rv), mpi_type_v<RecvScalar>, src,
             tag, comm, MPI_STATUS_IGNORE);
  }
}
} // namespace KokkosComm::Impl