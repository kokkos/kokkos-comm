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

#include <Kokkos_Core.hpp>

namespace KokkosComm {

namespace Impl {

// fallback - most types are not a KokkosComm transport
template <typename T>
struct is_transport : public std::false_type {};
}  // namespace Impl

template <typename T>
concept KokkosView = Kokkos::is_view_v<T>;

template <typename T>
concept KokkosExecutionSpace = Kokkos::is_execution_space_v<T>;

// TODO: a function that takes something convertible to a handle and returns void
template <typename T>
concept Dispatch = true;

template <typename T>
concept Transport = KokkosComm::Impl::is_transport<T>::value;

}  // namespace KokkosComm
