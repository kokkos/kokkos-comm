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

#include "impl/KokkosComm_mpi_isend_deepcopy.hpp"
#include "impl/KokkosComm_mpi_isend_datatype.hpp"
#include "KokkosComm_mpi.hpp"

namespace KokkosComm {

// low-level API
template <KokkosView SendView>
void isend(const SendView &sv, int dest, int tag, MPI_Comm comm, MPI_Request &req) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::isend");

  if (KokkosComm::is_contiguous(sv)) {
    using SendScalar = typename SendView::non_const_value_type;
    MPI_Isend(KokkosComm::data_handle(sv), KokkosComm::span(sv), Impl::mpi_type_v<SendScalar>, dest, tag, comm, &req);
  } else {
    throw std::runtime_error("only contiguous views supported for low-level isend");
  }
  Kokkos::Tools::popRegion();
}

namespace Impl {

// isend implementation for Mpi
template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
struct Isend<SendView, ExecSpace, Mpi> {
  Isend(Handle<ExecSpace, Mpi> &h, const SendView &sv, int src, int tag) {
    // Impl::isend_deepcopy(h, sv, src, tag);
    Impl::isend_datatype(h, sv, src, tag);
  }
};

}  // namespace Impl

}  // namespace KokkosComm
