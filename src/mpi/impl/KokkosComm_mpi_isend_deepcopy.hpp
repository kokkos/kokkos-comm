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

template <typename HandleTy, KokkosView SendView>
void isend_deepcopy(HandleTy &h, SendView &sv, int src, int tag) {
  // h.impl_track_view(sv);

  if (KokkosComm::is_contiguous(sv)) {
    h.impl_add_pre_comm_fence();  // make sure any work producing the send buffer is done

    h.impl_add_comm([&h, sv, src, tag]() {
      using Scalar = typename SendView::non_const_value_type;
      MPI_Request req;
      MPI_Isend(KokkosComm::data_handle(sv), KokkosComm::span(sv), mpi_type_v<Scalar>, src, tag, h.mpi_comm(), &req);
      h.impl_track_mpi_request(req);
    });

  } else {
    using ScratchView   = contiguous_view_t<SendView>;
    using ScratchScalar = typename ScratchView::non_const_value_type;
    auto scratch        = std::make_shared<ScratchView>();

    h.impl_add_alloc([&h, scratch, sv]() { *scratch = allocate_contiguous_for(h.space(), "TODO", sv); });

    h.impl_add_pre_copy([&h, scratch, sv]() { Kokkos::deep_copy(h.space(), *scratch, sv); });

    h.impl_add_pre_comm_fence();  // added a copy, so fence before communication

    h.impl_add_comm([&h, scratch, src, tag]() {
      MPI_Request req;
      MPI_Isend(KokkosComm::data_handle(*scratch), KokkosComm::span(*scratch), mpi_type_v<ScratchScalar>, src, tag,
                h.mpi_comm(), &req);
      h.impl_track_mpi_request(req);
    });
  }
}

}  // namespace KokkosComm::Impl
