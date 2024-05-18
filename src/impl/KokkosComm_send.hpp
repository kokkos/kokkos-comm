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

#include "KokkosComm_mpi.hpp"
#include "KokkosComm_pack_traits.hpp"
#include "KokkosComm_comm_mode.hpp"

namespace KokkosComm::Impl {

template <KokkosView SendView>
void send(SendView sv, int dest, int tag, Communicator comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::send");
  using KCT = typename KokkosComm::Traits<SendView>;

  if (KCT::is_contiguous(sv)) {
    comm.send(sv, dest, tag);
  } else {
    throw std::runtime_error("only contiguous views supported for low-level send");
  }
  Kokkos::Tools::popRegion();
}

template <CommMode SendMode = CommMode::Default, KokkosView SendView>
void send(KokkosExecutionSpace auto const &space, SendView sv, int dest, int tag, Communicator comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::send");
  using Packer = typename KokkosComm::PackTraits<SendView>::packer_type;

  if (KokkosComm::PackTraits<SendView>::needs_pack(sv)) {
    auto args = Packer::pack(space, sv);
    space.fence();
    comm.send(args.view, dest, tag);
  } else {
    comm.send(sv, dest, tag);
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::Impl
