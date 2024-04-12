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

#include "impl/KokkosComm_concepts.hpp"
#include "KokkosComm_pack_traits.hpp"
#include "KokkosComm_traits.hpp"

// impl
#include "KokkosComm_include_mpi.hpp"
#include "KokkosComm_types.hpp"

namespace KokkosComm::Impl {
template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView,
          NonContig SNC = DefaultNonContig<ExecSpace, SendView>, NonContig RNC = DefaultNonContig<ExecSpace, RecvView>>
void reduce(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Op op, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::reduce");

  const int rank = [=]() -> int {
    int _r;
    MPI_Comm_rank(comm, &_r);
    return _r;
  }();

  // This doesn't work directly with the datatype engine

  if (root == rank) {
    Ctx sctx = SNC::pre_send(space, sv);  // FIXME: terrible name
    Ctx rctx = RNC::pre_recv(space, rv);  // FIXME: terrible name
    space.fence();

    if (sctx.mpi_args.size() != rctx.mpi_args.size()) {
      throw std::logic_error("internal error");  // FIXME
    }

    for (size_t ai = 0; ai < sctx.mpi_args.size(); ++ai) {
      Ctx::MpiArgs &sargs = sctx.mpi_args[ai];
      Ctx::MpiArgs &rargs = rctx.mpi_args[ai];
      MPI_Reduce(sargs.buf, rargs.buf, sargs.count, sargs.datatype, op, root, comm);
    }
    RNC::post_recv(space, rv, rctx);
    space.fence();

  } else {
    Ctx sctx = SNC::pre_send(space, sv);  // FIXME: terrible name
    space.fence();

    for (size_t ai = 0; ai < sctx.mpi_args.size(); ++ai) {
      Ctx::MpiArgs &sargs = sctx.mpi_args[ai];
      MPI_Reduce(sargs.buf, rv.data() /*shouldn't matter*/, sargs.count, sargs.datatype, op, root, comm);
    }
  }

  Kokkos::Tools::popRegion();
}
}  // namespace KokkosComm::Impl
