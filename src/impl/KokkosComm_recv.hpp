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

#include "KokkosComm_concepts.hpp"
#include "KokkosComm_pack_traits.hpp"
#include "KokkosComm_traits.hpp"

// impl
#include "KokkosComm_include_mpi.hpp"

namespace KokkosComm::Impl {
template <KokkosExecutionSpace ExecSpace, KokkosView RecvView, NonContigSendRecv NC = DefaultNonContigSendRecv<ExecSpace, RecvView>>
void recv(const ExecSpace &space, RecvView &rv, int src, int tag, MPI_Comm comm) {
  using KCT = KokkosComm::Traits<RecvView>;
  Kokkos::Tools::pushRegion("KokkosComm::Impl::recv");

  space.fence("recv fence before checking view properties");

  // I think it's okay to use the same tag for all messages here due to
  // non-overtaking of messages that match the same recv
  CtxBufCount ctx = NC::pre_recv(space, rv);  // FIXME: terrible name
  space.fence();
  for (const CtxBufCount::MpiArgs &args : ctx.mpi_args) {
    MPI_Recv(args.buf, args.count, args.datatype, src, tag, comm, MPI_STATUS_IGNORE);
#if 0
    {
      std::stringstream ss;
      ss << __FILE__ << ":" << __LINE__ << ": MPI_Recv: @" << args.buf << " count=" << args.count;
      for (int i = 0; i < args.count; ++i) {
        ss << " " << reinterpret_cast<float*>(args.buf)[i];
      }
      ss << "\n";
      std::cerr << ss.str();
    }
#endif
  }
  NC::post_recv(space, rv, ctx);  // FIXME: terrible name
  space.fence();

#if 0
  if constexpr (RecvView::rank == 1) {
      std::stringstream ss;
      ss << __FILE__ << ":" << __LINE__ << ": MPI_Recv: @" << rv.data() << " rv.extent(0)=" << rv.extent(0);
      for (size_t i = 0; i < rv.extent(0); ++i) {
        ss << " " << rv[i];
      }
      ss << "\n";
      std::cerr << ss.str();
    }
#endif

  Kokkos::Tools::popRegion();
}
}  // namespace KokkosComm::Impl
