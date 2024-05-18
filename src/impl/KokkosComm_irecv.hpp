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

#include <memory>

#include <Kokkos_Core.hpp>

#include "KokkosComm_mpi.hpp"
#include "KokkosComm_pack_traits.hpp"
#include "KokkosComm_traits.hpp"

namespace KokkosComm::Impl {

// low-level API
template <KokkosView RecvView>
Request irecv(RecvView rv, int src, int tag, Communicator comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::irecv");

  using KCT = KokkosComm::Traits<RecvView>;

  if (KCT::is_contiguous(rv)) {
    return comm.irecv(rv, src, tag);
  } else {
    throw std::runtime_error("Only contiguous irecv viewsupported");
  }

  Kokkos::Tools::popRegion();
}
}  // namespace KokkosComm::Impl
