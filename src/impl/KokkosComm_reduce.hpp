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

#include "KokkosComm_pack_traits.hpp"
#include "KokkosComm_traits.hpp"
#include "KokkosComm_communicator.hpp"

// impl
#include "KokkosComm_include_mpi.hpp"
#include "KokkosComm_types.hpp"

namespace KokkosComm::Impl {
template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void reduce(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Op op, int root,
            KokkosComm::Communicator comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::reduce");

  using SendPacker = typename KokkosComm::PackTraits<SendView>::packer_type;
  using RecvPacker = typename KokkosComm::PackTraits<RecvView>::packer_type;

  if (KokkosComm::PackTraits<SendView>::needs_pack(sv)) {
    auto sendArgs = SendPacker::pack(space, sv);
    space.fence();
    if ((root == comm.rank()) && KokkosComm::PackTraits<RecvView>::needs_unpack(rv)) {
      auto recvArgs = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence();
      comm.reduce(sendArgs.view, recvArgs.view, op, root);
      RecvPacker::unpack_into(space, rv, recvArgs.view);
    } else {
      space.fence();
      comm.reduce(sendArgs.view, rv, op, root);
    }
  } else {
    if ((root == comm.rank()) && KokkosComm::PackTraits<RecvView>::needs_unpack(rv)) {
      auto recvArgs = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence();
      comm.reduce(sv, recvArgs.view, op, root);
      RecvPacker::unpack_into(space, rv, recvArgs.view);
    } else {
      space.fence();
      comm.reduce(sv, rv, op, root);
    }
  }

  Kokkos::Tools::popRegion();
}
}  // namespace KokkosComm::Impl
