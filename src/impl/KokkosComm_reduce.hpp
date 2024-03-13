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

// impl
#include "KokkosComm_include_mpi.hpp"
#include "KokkosComm_types.hpp"

namespace KokkosComm::Impl {
template <typename SendView, typename RecvView, typename ExecSpace>
void reduce(const ExecSpace &space, const SendView &sv, const RecvView &rv,
            MPI_Op op, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::reduce");

  const int rank = [=]() -> int {
    int _r;
    MPI_Comm_rank(comm, &_r);
    return _r;
  }();

  using SendPacker = typename KokkosComm::PackTraits<SendView>::packer_type;
  using RecvPacker = typename KokkosComm::PackTraits<RecvView>::packer_type;

  if (KokkosComm::PackTraits<SendView>::needs_pack(sv)) {
    auto sendArgs = SendPacker::pack(space, sv);
    space.fence();
    if ((root == rank) && KokkosComm::PackTraits<RecvView>::needs_unpack(rv)) {
      auto recvArgs = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence();
      MPI_Reduce(sendArgs.view.data(), recvArgs.view.data(), sendArgs.count,
                 sendArgs.datatype, op, root, comm);
      RecvPacker::unpack_into(space, rv, recvArgs.view);
    } else {
      space.fence();
      MPI_Reduce(sendArgs.view.data(), rv.data(), sendArgs.count,
                 sendArgs.datatype, op, root, comm);
    }
  } else {
    using SendScalar = typename SendView::value_type;
    if ((root == rank) && KokkosComm::PackTraits<RecvView>::needs_unpack(rv)) {
      auto recvArgs = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence();
      MPI_Reduce(sv.data(), recvArgs.view.data(), sv.span(),
                 mpi_type_v<SendScalar>, op, root, comm);
      RecvPacker::unpack_into(space, rv, recvArgs.view);
    } else {
      space.fence();
      MPI_Reduce(sv.data(), rv.data(), sv.span(), mpi_type_v<SendScalar>, op,
                 root, comm);
    }
  }

  Kokkos::Tools::popRegion();
}
}  // namespace KokkosComm::Impl