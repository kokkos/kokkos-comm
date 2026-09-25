// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <KokkosComm/concepts.hpp>

namespace KokkosComm {
namespace Experimental {

/// The RCCL communication space.
struct RcclSpace {
  using communication_space = RcclSpace;
  using communicator_type   = ncclComm_t;
  using request_type        = hipEvent_t;
  using datatype_type       = ncclDataType_t;
  using reduction_op_type   = ncclRedOp_t;
  using size_type           = int;
  using rank_type           = int;
};

}  // namespace Experimental

// KokkosComm::RcclSpace is a KokkosComm::CommunicationSpace
template <>
struct Impl::is_communication_space<Experimental::RcclSpace> : public std::true_type {};

}  // namespace KokkosComm
