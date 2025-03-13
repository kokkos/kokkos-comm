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

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>

#include "impl/pack_traits.hpp"
#include "impl/types.hpp"

namespace KokkosComm::mpi {

template <KokkosView SendView, KokkosView RecvView>
void reduce(const SendView &sv, const RecvView &rv, MPI_Op op, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::reduce");

  if (KokkosComm::is_contiguous(sv) && KokkosComm::is_contiguous(rv)) {
    using SendScalar = typename SendView::non_const_value_type;
    MPI_Reduce(KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), KokkosComm::span(sv),
               KokkosComm::Impl::mpi_type_v<SendScalar>, op, root, comm);
  } else {
    Kokkos::Tools::popRegion();
    throw std::runtime_error("only contiguous views supported for low-level reduce");
  }
  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void reduce(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Op op, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::reduce");

  const int rank = [=]() -> int {
    int _r;
    MPI_Comm_rank(comm, &_r);
    return _r;
  }();

  using SendPacker = typename KokkosComm::PackTraits<SendView>::packer_type;
  using RecvPacker = typename KokkosComm::PackTraits<RecvView>::packer_type;

  if (!KokkosComm::is_contiguous(sv)) {
    auto sendArgs = SendPacker::pack(space, sv);
    if ((root == rank) && !KokkosComm::is_contiguous(rv)) {
      auto recvArgs = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence("fence allocation before MPI call");
      MPI_Reduce(KokkosComm::data_handle(sendArgs.view), KokkosComm::data_handle(recvArgs.view), sendArgs.count,
                 sendArgs.datatype, op, root, comm);
      RecvPacker::unpack_into(space, rv, recvArgs.view);
    } else {
      space.fence("fence packing before MPI call");
      MPI_Reduce(KokkosComm::data_handle(sendArgs.view), KokkosComm::data_handle(rv), sendArgs.count, sendArgs.datatype,
                 op, root, comm);
    }
  } else {
    using SendScalar = typename SendView::value_type;
    if ((root == rank) && !KokkosComm::is_contiguous(rv)) {
      auto recvArgs = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence("fence allocation before MPI call");
      MPI_Reduce(KokkosComm::data_handle(sv), KokkosComm::data_handle(recvArgs.view), KokkosComm::span(sv),
                 KokkosComm::Impl::mpi_type_v<SendScalar>, op, root, comm);
      RecvPacker::unpack_into(space, rv, recvArgs.view);
    } else {
      space.fence("fence space before MPI call");
      MPI_Reduce(KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), KokkosComm::span(sv),
                 KokkosComm::Impl::mpi_type_v<SendScalar>, op, root, comm);
    }
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::mpi
