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

#include "KokkosComm_contiguous.hpp"
#include "KokkosComm_types.hpp"

namespace KokkosComm::Impl {

template <typename HandleTy, KokkosView RecvView>
void irecv_datatype(HandleTy &h, RecvView rv, int src, int tag) {
  h.impl_add_pre_comm_fence();

  h.impl_add_comm([&h, rv, src, tag]() {
    Req<Mpi> req;
    MPI_Irecv(KokkosComm::data_handle(rv), 1, view_mpi_type(rv), src, tag, h.mpi_comm(), &req.mpi_req());
    h.impl_track_req(req);
  });
}

}  // namespace KokkosComm::Impl
