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
          NonContigReduce NC = DefaultNonContigReduce<ExecSpace, SendView, RecvView>>
void reduce(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Op op, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::reduce");

  CtxReduce ctx = NC::pre_reduce(space, sv, rv);
  if (ctx.pre_uses_space()) space.fence();
  for (const auto &args : ctx.mpi_args) {
    MPI_Reduce(args.sbuf, args.rbuf, args.count, args.datatype, op, root, comm);
  }
  NC::post_reduce(space, sv, rv, ctx);

  Kokkos::Tools::popRegion();
}
}  // namespace KokkosComm::Impl
