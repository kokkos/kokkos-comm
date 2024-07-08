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

#include "KokkosComm_comm_modes.hpp"

#include <Kokkos_Core.hpp>

namespace KokkosComm {

template <typename T>
concept KokkosView = Kokkos::is_view_v<T>;

template <typename T>
concept KokkosExecutionSpace = Kokkos::is_execution_space_v<T>;

template <typename T>
struct is_communication_mode : std::false_type {};

template <>
struct is_communication_mode<StandardCommMode> : std::true_type {};

template <>
struct is_communication_mode<SynchronousCommMode> : std::true_type {};

template <>
struct is_communication_mode<ReadyCommMode> : std::true_type {};

template <typename T>
inline constexpr bool is_communication_mode_v = is_communication_mode<T>::value;

template <typename T>
concept CommunicationMode = KokkosComm::is_communication_mode_v<T>;

}  // namespace KokkosComm
