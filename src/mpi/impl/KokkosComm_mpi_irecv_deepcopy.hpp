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
void irecv_deepcopy(HandleTy &h, RecvView rv, int src, int tag) {
  if (KokkosComm::is_contiguous(rv)) {
    h.impl_add_pre_comm_fence();

    h.impl_add_comm([&h, rv, src, tag]() {
      using Scalar = typename RecvView::non_const_value_type;
      MPI_Request req;
      MPI_Irecv(KokkosComm::data_handle(rv), KokkosComm::span(rv), mpi_type_v<Scalar>, src, tag, h.mpi_comm(), &req);
      h.impl_track_mpi_request(req);
    });
  } else {
    using ScratchView   = contiguous_view_t<RecvView>;
    using ScratchScalar = typename ScratchView::non_const_value_type;
    auto scratch        = std::make_shared<ScratchView>();
    h.impl_add_alloc([&h, rv, scratch]() { resize_contiguous_for(h.space(), *scratch, rv); });

    h.impl_add_pre_comm_fence();

    // add scratch allocation and send during comm phase
    h.impl_add_comm([&h, scratch, src, tag]() {
      // recv into scratch buffer
      MPI_Request req;
      MPI_Irecv(KokkosComm::data_handle(*scratch), KokkosComm::span(*scratch), mpi_type_v<ScratchScalar>, src, tag,
                h.mpi_comm(), &req);
      h.impl_track_mpi_request(req);
    });
    // register callback to copy scratch buffer into rv after MPI_Wait
    h.impl_add_post_wait([&h, rv, scratch]() { Kokkos::deep_copy(h.space(), rv, *scratch); });
  }
}

}  // namespace KokkosComm::Impl
