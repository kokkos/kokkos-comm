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

#include <KokkosComm/traits.hpp>
#include <KokkosComm/mpi/impl/pack_traits.hpp>
#include <KokkosComm/mpi/impl/include_mpi.hpp>
#include <KokkosComm/mpi/impl/types.hpp>

#include <Kokkos_Core.hpp>

namespace KokkosComm::mpi {

template <KokkosView SendView, KokkosView RecvView>
void reduce(const SendView &sv, const RecvView &rv, MPI_Op op, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::reduce");
  using SPT = KokkosComm::PackTraits<SendView>;
  using RPT = KokkosComm::PackTraits<RecvView>;

  if (SPT::is_contiguous(sv) && RPT::is_contiguous(rv)) {
    using SendScalar = typename SendView::non_const_value_type;
    MPI_Reduce(SPT::data_handle(sv), RPT::data_handle(rv), SPT::span(sv), KokkosComm::Impl::mpi_type_v<SendScalar>, op,
               root, comm);
  } else {
    throw std::runtime_error("only contiguous views supported for low-level reduce");
  }
  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void reduce(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Op op, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::reduce");

  const int rank = [=]() -> int {
    int _r;
    MPI_Comm_rank(comm, &_r);
    return _r;
  }();

  using SendPacker = typename KokkosComm::PackTraits<SendView>::packer_type;
  using RecvPacker = typename KokkosComm::PackTraits<RecvView>::packer_type;

  if (!KokkosComm::is_contiguous(sv)) {
    auto sendArgs = SendPacker::pack(space, sv);
    space.fence();
    if ((root == rank) && !KokkosComm::is_contiguous(rv)) {
      auto recvArgs = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence();
      MPI_Reduce(sendArgs.view.data(), recvArgs.view.data(), sendArgs.count, sendArgs.datatype, op, root, comm);
      RecvPacker::unpack_into(space, rv, recvArgs.view);
    } else {
      space.fence();
      MPI_Reduce(sendArgs.view.data(), rv.data(), sendArgs.count, sendArgs.datatype, op, root, comm);
    }
  } else {
    using SendScalar = typename SendView::value_type;
    if ((root == rank) && !KokkosComm::is_contiguous(rv)) {
      auto recvArgs = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence();
      MPI_Reduce(sv.data(), recvArgs.view.data(), sv.span(), KokkosComm::Impl::mpi_type_v<SendScalar>, op, root, comm);
      RecvPacker::unpack_into(space, rv, recvArgs.view);
    } else {
      space.fence();
      MPI_Reduce(sv.data(), rv.data(), sv.span(), KokkosComm::Impl::mpi_type_v<SendScalar>, op, root, comm);
    }
  }

  Kokkos::Tools::popRegion();
}
}  // namespace KokkosComm::mpi
