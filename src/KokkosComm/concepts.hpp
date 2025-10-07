// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

namespace KokkosComm {

namespace Impl {

// Fallback: types are not a KokkosComm communication space by default
template <typename T>
struct is_communication_space : public std::false_type {};

}  // namespace Impl

template <typename T>
concept KokkosView = Kokkos::is_view_v<T>;

template <typename T>
concept KokkosExecutionSpace = Kokkos::is_execution_space_v<T>;

template <typename T>
concept CommunicationSpace = KokkosComm::Impl::is_communication_space<T>::value;

}  // namespace KokkosComm
