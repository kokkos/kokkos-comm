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

#include <type_traits>

// See section 3.4 of the MPI standard for a complete specification.

namespace KokkosComm::mpi {
// Standard mode: MPI implementation decides whether outgoing messages will
// be buffered. Send operations can be started whether or not a matching
// receive has been started. They may complete before a matching receive is
// started. Standard mode is non-local: successful completion of the send
// operation may depend on the occurrence of a matching receive.
struct CommModeStandard {};

// Ready mode: Send operations may be started only if the matching receive is
// already started.
struct CommModeReady {};

// Synchronous mode: Send operations complete successfully only if a matching
// receive is started, and the receive operation has started to receive the
// message sent.
struct CommModeSynchronous {};

// Default mode: lets the user override the send operations behavior at
// compile-time. E.g., this can be set to mode "Synchronous" for debug
// builds by defining KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE.
#ifdef KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE
using DefaultCommMode = CommModeSynchronous;
#else
using DefaultCommMode = CommModeStandard;
#endif

template <typename T>
struct is_communication_mode : std::false_type {};

template <>
struct is_communication_mode<CommModeStandard> : std::true_type {};

template <>
struct is_communication_mode<CommModeSynchronous> : std::true_type {};

template <>
struct is_communication_mode<CommModeReady> : std::true_type {};

template <typename T>
inline constexpr bool is_communication_mode_v = is_communication_mode<T>::value;

template <typename T>
concept CommunicationMode = is_communication_mode_v<T>;

}  // namespace KokkosComm::mpi