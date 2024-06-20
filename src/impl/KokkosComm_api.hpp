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

namespace KokkosComm::Impl {

enum class Api { Irecv, Isend };

// catch-all: no transports implement any APIs
template <typename Transport, Impl::Api API>
struct api_avail : public std::false_type {};

template <typename Transport, Impl::Api API>
constexpr bool api_avail_v = api_avail<Transport, API>::value;

}  // namespace KokkosComm::Impl
