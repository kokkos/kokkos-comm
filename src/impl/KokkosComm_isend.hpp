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

#include <iostream>

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include "KokkosComm_request.hpp"

// impl
#include "KokkosComm_mdspan.hpp"
#include "KokkosComm_pack.hpp"

namespace KokkosComm::Impl {

template <typename Span, typename ExecSpace>
KokkosComm::Req isend(const ExecSpace &space, const Span &ss, int dest, int tag,
                      MPI_Comm comm) {
  KokkosComm::Req req;

  using KCT = KokkosComm::Traits<Span>;

  if (KCT::needs_pack(ss)) {
    using PackedScalar = typename KCT::packed_view_type::value_type;
    auto packed = allocate_packed_for(space, "packed", ss);
    pack(space, packed, ss);
    space.fence();
    MPI_Isend(KCT::data_handle(packed),
              KCT::span(packed) * sizeof(PackedScalar), MPI_PACKED, dest, tag,
              comm, &req.mpi_req());
    req.keep_until_wait(packed);
  } else {
    using SendScalar = typename Span::value_type;
    MPI_Isend(KCT::data_handle(ss), KCT::span(ss), mpi_type_v<SendScalar>, dest,
              tag, comm, &req.mpi_req());
    if (KCT::is_reference_counted()) {
      req.keep_until_wait(ss);
    }
  }
  return req;
}

} // namespace KokkosComm::Impl