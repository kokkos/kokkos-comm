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

#include <Kokkos_Core.hpp>

namespace KokkosComm::Impl {

template <typename T>
concept KokkosView = Kokkos::is_view_v<T>;

template <typename T>
concept KokkosExecutionSpace = Kokkos::is_execution_space_v<T>;

template <typename Fn>
concept Invokable = std::is_invocable_v<Fn>;

template <typename T>
concept NonContigSendRecv = !std::is_void_v<T>;  // FIXME: placeholder

template <typename T>
concept NonContigReduce = !std::is_void_v<T>;  // FIXME: placeholder

template <typename T>
concept NonContigAlltoall = !std::is_void_v<T>;  // FIXME: placeholder

}  // namespace KokkosComm::Impl
