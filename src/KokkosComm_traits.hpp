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

/*! \brief Defines a common interface for Kokkos::View-like types
    \file KokkosComm_traits.hpp
*/

#pragma once

#include "KokkosComm_concepts.hpp"

namespace KokkosComm {

template <typename T>
struct Traits {
  static_assert(std::is_void_v<T>, "KokkosComm::Traits not specialized for type");
};

/*! \brief This can be specialized to do custom behavior for a particular view*/
template <KokkosView View>
struct Traits<View> {
  // product of extents is span
  static bool is_contiguous(const View &v) { return v.span_is_contiguous(); }

  static auto data_handle(const View &v) { return v.data(); }

  using non_const_packed_view_type =
      Kokkos::View<typename View::non_const_data_type, typename View::array_layout, typename View::memory_space>;
  using packed_view_type =
      Kokkos::View<typename View::data_type, typename View::array_layout, typename View::memory_space>;

  static size_t span(const View &v) { return v.span(); }

  static size_t extent(const View &v, const int i) { return v.extent(i); }
  static size_t stride(const View &v, const int i) { return v.stride(i); }

  static constexpr bool is_reference_counted() { return true; }

  static constexpr size_t rank() { return View::rank; }
};

}  // namespace KokkosComm
