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

#include <Kokkos_Core_fwd.hpp>  // Kokkos::Cuda
#include <nccl.h>

#include <type_traits>

namespace KokkosComm::Experimental {

struct Nccl {
  using communication_space = Nccl;
  using execution_space     = Kokkos::Cuda;
  using datatype_type       = ncclDataType_t;
  using reduction_op_type   = ncclRedOp_t;
};

// Nccl is a KokkosComm::CommunicationSpace
template <>
struct KokkosComm::Impl::is_communication_space<Nccl> : public std::true_type {};

}  // namespace KokkosComm::Experimental
