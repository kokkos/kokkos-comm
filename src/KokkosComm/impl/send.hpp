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

#include <KokkosComm/fwd.hpp>
#include <KokkosComm/concepts.hpp>

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <KokkosComm/nccl/handle.hpp>
#include <KokkosComm/nccl/send.hpp>
#endif

#include <Kokkos_Core.hpp>

namespace KokkosComm::Impl {

#if defined(KOKKOSCOMM_ENABLE_NCCL)
template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::Cuda, CommunicationSpace CommSpace = Nccl>
struct Send {
  static auto execute(Handle<ExecSpace, CommSpace> &h, const SendView sv, int dst) -> Req<CommSpace> {
    Experimental::nccl::Impl::send(h.space(), sv, dst, h.comm());
    return Req<CommSpace>(h.space().cuda_stream());
  }
};
#endif

}  // namespace KokkosComm::Impl
