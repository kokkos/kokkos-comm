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
#include "KokkosComm_concepts.hpp"

namespace KokkosComm::Impl {
template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView,
          NonContigAlltoall NC = DefaultNonContigAlltoall<ExecSpace, SendView, RecvView>>
void alltoall(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::alltoall");

  // FIXME: check some relative sizes of sv and rv, etc
  // FIXME: debug only
  {}

  int size;
  MPI_Comm_size(comm, &size);
  const int scount = sv.size() / size;
  const int rcount = rv.size() / size;

  CtxAlltoall ctx = NC::pre_alltoall(space, sv, rv, scount, rcount);

  if (ctx.pre_uses_space()) {
    space.fence("ctx fence before MPI_Alltoall");
  }
  for (const CtxAlltoall::MpiArgs &args : ctx.mpi_args) {
    MPI_Alltoall(args.sbuf, args.scount, args.stype, args.rbuf, args.rcount, args.rtype, comm);
  }
  NC::post_alltoall(space, rv, ctx);

  Kokkos::Tools::popRegion();
}

// in-place alltoall
template <KokkosExecutionSpace ExecSpace, KokkosView View,
          NonContigAlltoall NC = DefaultNonContigAlltoall<ExecSpace, View, View>>
void alltoall(const ExecSpace &space, const View &v, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::alltoall");

  int size;
  MPI_Comm_size(comm, &size);
  const int rcount = v.size() / size;

  CtxAlltoall ctx = NC::pre_alltoall_inplace(space, v, rcount);
  space.fence();
  for (const CtxAlltoall::MpiArgs &args : ctx.mpi_args) {
    MPI_Alltoall(args.sbuf, args.scount, args.stype, args.rbuf, args.rcount, args.rtype, comm);
  }
  NC::post_alltoall_inplace(space, v, ctx);

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::Impl
