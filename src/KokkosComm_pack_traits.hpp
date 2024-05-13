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

#include "KokkosComm_traits.hpp"

#include "KokkosComm_concepts.hpp"
#include "KokkosComm_packer.hpp"

/*! \brief Defines a common interface for packing and unpacking
   Kokkos::View-like types \file KokkosComm_traits.hpp
*/

namespace KokkosComm {

template <typename T>
struct PackTraits {
  static_assert(std::is_void_v<T>, "KokkosComm::PackTraits not specialized for type");
};

/*! \brief This can be specialized to do custom behavior for a particular view*/
template <KokkosView View>
struct PackTraits<View> {
  using packer_type = Impl::Packer::DeepCopy<View>;

  static bool needs_unpack(const View &v) { return !Traits<View>::is_contiguous(v); }
  static bool needs_pack(const View &v) { return !Traits<View>::is_contiguous(v); }
};

}  // namespace KokkosComm
