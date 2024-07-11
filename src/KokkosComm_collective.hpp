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

#include <utility>

#include <Kokkos_Core.hpp>

#include "KokkosComm_fwd.hpp"
#include "KokkosComm_concepts.hpp"

namespace KokkosComm {

template <KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
void barrier(Handle<ExecSpace, CommSpace> &&h) {
  Impl::Barrier<ExecSpace, CommSpace>{std::forward<Handle<ExecSpace, CommSpace>>(h)};
}

}  // namespace KokkosComm
