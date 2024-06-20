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

#include "KokkosComm_concepts.hpp"
#include "KokkosComm_traits.hpp"
#include "KokkosComm_types.hpp"

#include "impl/KokkosComm_mpi_irecv_deepcopy.hpp"
#include "impl/KokkosComm_mpi_irecv_datatype.hpp"

namespace KokkosComm {

// low-level API
template <KokkosView RecvView>
void irecv(RecvView &rv, int src, int tag, MPI_Comm comm, MPI_Request &req) {
  Kokkos::Tools::pushRegion("KokkosComm::irecv");

  if (KokkosComm::is_contiguous(rv)) {
    using RecvScalar = typename RecvView::value_type;
    MPI_Irecv(KokkosComm::data_handle(rv), KokkosComm::span(rv), Impl::mpi_type_v<RecvScalar>, src, tag, comm, &req);
  } else {
    throw std::runtime_error("Only contiguous irecv viewsupported");
  }

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void Mpi::irecv(Mpi::Handle<ExecSpace> &h, const RecvView &rv, int src, int tag) {
  // Impl::irecv_deepcopy(h, rv, src, tag);
  Impl::irecv_datatype(h, rv, src, tag);
}

}  // namespace KokkosComm
