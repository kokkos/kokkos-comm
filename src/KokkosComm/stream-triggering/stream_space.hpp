// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <mpi.h>

#include <stream-triggering.h>

#include <KokkosComm/concepts.hpp>

namespace KokkosComm {
  namespace Experimental {

/// The MPI communication space.
struct StreamSpace {
  using communication_space = MpiSpace;
  using communicator_type   = MPI_Comm;
  using request_type        = MPI_Request;
  using datatype_type       = MPI_Datatype;
  using reduction_op_type   = MPI_Op;
  using size_type           = int;
  using rank_type           = int;
};

}  // namespace Experimental
    
// KokkosComm::StreamSpace is a KokkosComm::CommunicationSpace
template <>
struct Impl::is_communication_space<Experimental::StreamSpace> : public std::true_type {};

}  // namespace KokkosComm
