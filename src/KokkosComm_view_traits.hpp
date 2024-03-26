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

#include "KokkosComm_mdspan.hpp"

#include "KokkosComm_concepts.hpp"

namespace KokkosComm {

template <typename T>
struct Traits {
  static_assert(std::is_void_v<T>,
                "KokkosComm::Traits not specialized for type");
};

/*! \brief This can be specialized to do custom behavior for a particular view*/
template <KokkosView View>
struct Traits<View> {
  using pointer_type = typename View::pointer_type;
  using scalar_type  = typename View::non_const_value_type;

  static bool is_contiguous(const View &v) { return v.span_is_contiguous(); }

  static pointer_type data_handle(const View &v) { return v.data(); }

  static size_t span(const View &v) { return v.span(); }

  static size_t extent(const View &v, const int i) { return v.extent(i); }

  static size_t stride(const View &v, const int i) { return v.stride(i); }

  static constexpr bool is_reference_counted() { return true; }

  static constexpr size_t rank() { return View::rank; }
};
#if KOKKOSCOMM_ENABLE_MDSPAN

template <Mdspan Span>
struct Traits<Span> {
  using pointer_type = typename Span::data_handle_type;
  using scalar_type  = typename Span::element_type;

  static bool is_contiguous(const Span &v) {
    size_t prod = 1;
    for (size_t i = 0; i < rank(); ++i) {
      prod *= v.extents().extent(i);
    }
    return prod == span(v);
  }
  static pointer_type data_handle(const Span &v) { return v.data_handle(); }

  // static size_t span(const packed_view_type &packed) { return packed.size();
  // }
  static size_t span(const Span &v) {
    std::array<typename Span::index_type, rank()> first, last;
    for (size_t i = 0; i < rank(); ++i) {
      first[i] = 0;
      last[i]  = v.extents().extent(i) - 1;
    }
    return &v[last] - &v[first] + sizeof(typename Span::value_type);
  }

  static size_t extent(const Span &v, const int i) {
    return v.extents().extent(i);
  }

  static size_t stride(const Span &v, const int i) { return v.stride(i); }

  static constexpr bool is_reference_counted() { return false; }

  static constexpr size_t rank() { return Span::extents_type::rank(); }
};

#endif  // KOKKOSCOMM_ENABLE_MDSPAN

template <typename T>
concept ViewLike = requires(T v) {
  { Traits<T>::is_contiguous(v) }
  ->std::convertible_to<bool>;
  { Traits<T>::data_handle(v) }
  ->std::convertible_to<typename Traits<T>::pointer_type>;
  { Traits<T>::span(v) }
  ->std::convertible_to<size_t>;
  { Traits<T>::extent(v, int{}) }
  ->std::convertible_to<size_t>;
  { Traits<T>::stride(v, int{}) }
  ->std::convertible_to<size_t>;
  { Traits<T>::is_reference_counted() }
  ->std::convertible_to<bool>;
  { Traits<T>::rank() }
  ->std::convertible_to<size_t>;
};

}  // namespace KokkosComm
