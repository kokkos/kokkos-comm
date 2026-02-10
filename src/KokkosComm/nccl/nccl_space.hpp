// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <nccl.h>
#include <cuda_runtime.h>

#include <KokkosComm/concepts.hpp>

namespace KokkosComm {
namespace Experimental {

/// The NCCL communication space.
struct NcclSpace {
  using communication_space = NcclSpace;
  using handle_type         = ncclComm_t;
  using request_type        = cudaEvent_t;
  using datatype_type       = ncclDataType_t;
  using reduction_op_type   = ncclRedOp_t;
  using rank_type           = int;
};

}  // namespace Experimental

// KokkosComm::NcclSpace is a KokkosComm::CommunicationSpace
template <>
struct Impl::is_communication_space<Experimental::NcclSpace> : public std::true_type {};

}  // namespace KokkosComm
