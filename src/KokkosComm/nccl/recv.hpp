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

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/nccl/impl/types.hpp>
#include <KokkosComm/nccl/impl/pack_traits.hpp>

#include <Kokkos_Core.hpp>
#include <nccl.h>

namespace KokkosComm::Experimental::nccl::Impl {

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void recv(const ExecSpace &space, RecvView &rv, int src, ncclComm_t comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::recv");

  if (KokkosComm::is_contiguous(rv)) {
    using RecvScalar = typename RecvView::value_type;
    ncclRecv(rv.data(), rv.span(), datatype_v<RecvScalar>, src, comm, space.cuda_stream());
  } else {
    using Packer = typename Impl::PackTraits<RecvView>::packer_type;
    auto args    = Packer::pack(space, rv);
    // TODO: consider using a private stream pool in order to avoid synchronizing the underlying stream (which may not
    // be empty and have in-flight communications we don't want to wait on)
    space.fence();  // make sure allocation is complete before receiving

    ncclRecv(KokkosComm::data_handle(args.view), args.count, args.datatype, src, comm, space.cuda_stream());
    Packer::unpack_into(space, rv, args.view);
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::Experimental::nccl::Impl
