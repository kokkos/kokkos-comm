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

namespace KokkosComm {

// Scoped enumeration to specify the communication mode of a sending operation.
// See section 3.4 of the MPI standard for a complete specification.
enum class CommMode {
  // Default mode: lets the user override the send operations behavior at compile-time. E.g. this can be set to mode
  // "Synchronous" for debug builds by defining KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE.
  Default,
  // Standard mode: MPI implementation decides whether outgoing messages will be buffered. Send operations can be
  // started whether or not a matching receive has been started. They may complete before a matching receive is started.
  // Standard mode is non-local: successful completion of the send operation may depend on the occurrence of a matching
  // receive.
  Standard,
  // Buffered mode: Send operation can be started whether or not a matching receive has been started. It may complete
  // before a matching receive is started. However, unlike the standard send, this operation is local, and its
  // completion does not depend on the occurrence of a matching receive. Thus, if a send is executed and no matching
  // receive is started, then MPI must buffer the outgoing message, so as to allow the send call to complete. An error
  // will occur if there is insufficient buffer space. The amount of available buffer space is controlled by the user
  // (see Section 3.6). Buffer allocation by the user may be required for the buffered mode to be effective.
  Buffered,
  // Ready mode: Send operations may be started only if the matching receive is already started.
  Ready,
  // Synchronous mode: Send operations complete successfully only if a matching receive is started, and the receive
  // operation has started to receive the message sent.
  Synchronous,
};

}  // namespace KokkosComm
