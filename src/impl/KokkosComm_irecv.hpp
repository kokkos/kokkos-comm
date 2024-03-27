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

namespace KokkosComm::Impl {

template <typename ExecSpace, typename RecvView, typename MpiArgs>
struct IrecvUnpacker {
  IrecvUnpacker(const ExecSpace &space, RecvView &rv, MpiArgs &args)
      : space_(space), rv_(rv), args_(args) {}

  void operator()() {
    Kokkos::Tools::pushRegion("KokkosComm::Impl::IrecvUnpacker::operator()");
    MpiArgs::packer_type::unpack_into(space_, rv_, args_.view);
    space_.fence();
    Kokkos::Tools::popRegion();
  }

  ExecSpace space_;
  RecvView rv_;
  MpiArgs args_;
};

/* FIXME: If RecvView is a Kokkos view, it can be a const ref
   same is true for an mdspan?
*/
template <typename RecvView, typename ExecSpace>
Req irecv(const ExecSpace &space, RecvView &rv, int src, int tag,
          MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::irecv");

  using KCT  = KokkosComm::Traits<RecvView>;
  using KCPT = KokkosComm::PackTraits<RecvView>;

  KokkosComm::Req req;

  if (KCPT::needs_unpack(rv)) {
    using Packer = typename KCPT::packer_type;
    using Args   = typename Packer::args_type;

    Args args = Packer::allocate_packed_for(space, "packed", rv);
    space.fence();
    MPI_Irecv(KCT::data_handle(args.view), args.count, args.datatype, src, tag,
              comm, &req.mpi_req());
    req.call_and_drop_at_wait(IrecvUnpacker{space, rv, args});

  } else {
    using RecvScalar = typename RecvView::value_type;
    MPI_Irecv(KCT::data_handle(rv), KCT::span(rv), mpi_type_v<RecvScalar>, src,
              tag, comm, &req.mpi_req());
    req.keep_until_wait(rv);
  }
  return req;

  Kokkos::Tools::popRegion();
}
}  // namespace KokkosComm::Impl