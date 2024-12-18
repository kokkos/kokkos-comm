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

#include <nccl.h>

namespace KokkosComm::Experimental::nccl::Impl {

template <typename Scalar>
ncclDataType_t datatype() {
  using T = std::decay_t<Scalar>;

  if constexpr (std::is_same_v<T, char>) {
    return ncclChar;
  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    return ncclInt8;
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    return ncclUint8;
  } else if constexpr (std::is_same_v<T, int>) {
    return ncclInt;
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return ncclInt32;
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return ncclUint32;
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return ncclInt64;
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return ncclUint64;
  } else if constexpr (std::is_same_v<T, float>) {
    return ncclFloat;
  } else if constexpr (std::is_same_v<T, double>) {
    return ncclDouble;
  } else {
    {
      static_assert(std::is_void_v<T>, "nccl_type not implemented");
      return ncclChar;  // unreachable
    }
  }
}

template <typename Scalar>
inline ncclDataType_t datatype_v = datatype<Scalar>();

};  // namespace KokkosComm::Experimental::nccl::Impl
