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

namespace KokkosComm::Impl {
template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView,
          NonContig SNC = DefaultNonContig<ExecSpace, SendView>, NonContig RNC = DefaultNonContig<ExecSpace, RecvView>>
void alltoall(const ExecSpace &space, const SendView &sv, const size_t sendCount, const RecvView &rv,
              const size_t recvCount, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::alltoall");

  // FIXME: only in debug builds
  {
    using ST = KokkosComm::Traits<SendView>;
    using RT = KokkosComm::Traits<RecvView>;

    int size;
    MPI_Comm_size(comm, &size);
    if (sendCount * size > ST::span(sv)) {
      std::stringstream ss;
      ss << "alltoall sendCount * communicator size (" << sendCount << " * " << size
         << ") is greater than send view size";
      throw std::runtime_error(ss.str());
    }
    if (recvCount * size > RT::span(rv)) {
      std::stringstream ss;
      ss << "alltoall recvCount * communicator size (" << recvCount << " * " << size
         << ") is greater than recv view size";
      throw std::runtime_error(ss.str());
    }
  }

  Ctx sctx = SNC::pre_send(space, sv);
  Ctx rctx = RNC::pre_recv(space, rv);

  if (sctx.mpi_args.size() != rctx.mpi_args.size()) {
    throw std::runtime_error(
        "alltoall: context mpi argument size mismatch between send and recv. This is a KokkosComm internal error, "
        "please report this.");
  }

  space.fence();
  for (size_t i = 0; i < sctx.mpi_args.size(); ++i) {
    Ctx::MpiArgs &sargs = sctx.mpi_args[i];
    Ctx::MpiArgs &rargs = rctx.mpi_args[i];
    MPI_Alltoall(sargs.buf, sargs.count, sargs.datatype, rargs.buf, rargs.count, rargs.datatype, comm);
  }
  RNC::post_recv(space, rv, rctx);

  Kokkos::Tools::popRegion();
}

// in-place alltoall
template <KokkosExecutionSpace ExecSpace, KokkosView View, NonContig NC = DefaultNonContig<ExecSpace, View>>
void alltoall(const ExecSpace &space, const View &v, const size_t recvCount, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::alltoall");

  // FIXME: only in debug builds
  {
    using KCT = KokkosComm::Traits<View>;
    int size;
    MPI_Comm_size(comm, &size);
    if (recvCount * size > KCT::span(v)) {
      std::stringstream ss;
      ss << "alltoall recvCount * communicator size (" << recvCount << " * " << size
         << ") is greater than recv view size";
      throw std::runtime_error(ss.str());
    }
  }

  Ctx ctx = NC::pre_send(space, v);
  space.fence();
  for (const Ctx::MpiArgs &args : ctx.mpi_args) {
    MPI_Alltoall(MPI_IN_PLACE, 0 /*ignored*/, MPI_BYTE /*ignored*/, args.buf, args.count, args.datatype, comm);
  }
  NC::post_recv(space, v, ctx);

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::Impl
