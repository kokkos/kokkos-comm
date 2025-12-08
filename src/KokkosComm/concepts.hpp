// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

namespace KokkosComm {
namespace Impl {

/// Fallback: most types are not a KokkosComm communication space
template <typename T>
struct is_communication_space : public std::false_type {};

/// Fallback: most types are not a KokkosComm reduction operator
template <typename T>
struct is_reduction_operator : public std::false_type {};

}  // namespace Impl

template <typename T>
concept KokkosView = Kokkos::is_view_v<T>;

template <typename T>
concept KokkosExecutionSpace = Kokkos::is_execution_space_v<T>;

template <typename T>
concept CommunicationSpace = requires {
  KokkosComm::Impl::is_communication_space<T>::value;
  typename T::communication_space;
  typename T::handle_type;
  typename T::request_type;
  typename T::datatype_type;
  typename T::reduction_op_type;
  typename T::rank_type;
};

template <typename T>
concept ReductionOperator = KokkosComm::Impl::is_reduction_operator<T>::value;

}  // namespace KokkosComm
