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

// src
#include "KokkosComm_traits.hpp"

template <typename Dst, typename Src, typename ExecSpace>
void pack(const ExecSpace &space, Dst &dst, const Src &src) {
  Kokkos::Tools::pushRegion("KokkosComm::pack");
  KokkosComm::Traits<Src>::pack(space, dst, src);
  Kokkos::Tools::popRegion();
}
