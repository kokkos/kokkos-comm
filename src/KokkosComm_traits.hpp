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
  static bool is_contiguous(const View &v) { return v.span_is_contiguous(); }

  static auto data_handle(const View &v) { return v.data(); }

  using non_const_packed_view_type =
      Kokkos::View<typename View::non_const_data_type,
                   typename View::array_layout, typename View::memory_space>;
  using packed_view_type =
      Kokkos::View<typename View::data_type, typename View::array_layout,
                   typename View::memory_space>;

  static size_t span(const View &v) { return v.span(); }

  static size_t extent(const View &v, const int i) { return v.extent(i); }
  static size_t stride(const View &v, const int i) { return v.stride(i); }

  template <typename ExecSpace>
  static void pack(const ExecSpace &space,
                   const non_const_packed_view_type &dst, const View &src) {
    Kokkos::deep_copy(space, dst, src);
  }

  template <typename ExecSpace>
  static void unpack(const ExecSpace &space, const View &dst,
                     const non_const_packed_view_type &src) {
    Kokkos::deep_copy(space, dst, src);
  }

  static constexpr bool is_reference_counted() {
    return !View::memory_traits::is_unmanaged;
  }

  static constexpr size_t rank() { return View::rank; }
};
#if KOKKOSCOMM_ENABLE_MDSPAN

template <Mdspan Span>
struct Traits<Span> {
  static auto data_handle(const Span &v) { return v.data_handle(); }

  using non_const_packed_view_type = std::vector<typename Span::value_type>;
  using packed_view_type           = std::vector<typename Span::value_type>;

  static auto data_handle(non_const_packed_view_type &v) { return v.data(); }

  static constexpr size_t rank() { return Span::extents_type::rank(); }

  static size_t extent(const Span &v, const int i) {
    return v.extents().extent(i);
  }
  static size_t stride(const Span &v, const int i) { return v.stride(i); }

  static size_t span(const packed_view_type &packed) { return packed.size(); }
  static size_t span(const Span &v) {
    std::array<typename Span::index_type, rank()> first, last;
    for (size_t i = 0; i < rank(); ++i) {
      first[i] = 0;
      last[i]  = v.extents().extent(i) - 1;
    }
    return &v[last] - &v[first] + sizeof(typename Span::value_type);
  }

  static bool is_contiguous(const Span &v) {
    size_t prod = 1;
    for (size_t i = 0; i < rank(); ++i) {
      prod *= v.extents().extent(i);
    }
    return prod == span(v);
  }

  template <typename ExecSpace>
  static void pack(const ExecSpace &space, non_const_packed_view_type &dst,
                   const Span &src) {
    using md_index = std::array<typename Span::index_type, rank()>;

    md_index index, ext;
    index = {0};
    for (size_t i = 0; i < rank(); ++i) {
      ext[i] = src.extents().extent(i);
    }
    size_t offset = 0;

    auto index_lt = [](const md_index &a, const md_index &b) -> bool {
      for (size_t i = 0; i < rank(); ++i) {
        if (!(a[i] < b[i])) {
          return false;
        }
      }
      return true;
    };

    auto increment = [&]() -> void {
      for (size_t i = 0; i < rank(); ++i) {
        ++index[i];
        if (index[i] >= ext[i] && i != rank() - 1 /* don't wrap final index*/) {
          index[i] = 0;
        } else {
          break;
        }
      }

      ++offset;
    };

    for (; index_lt(index, ext); increment()) {
      dst[offset] = src[index];
    }
  }

  template <typename ExecSpace>
  static void unpack(const ExecSpace &space, Span &dst,
                     const non_const_packed_view_type &src) {
    using md_index = std::array<typename Span::index_type, rank()>;

    md_index index, ext;
    index = {0};
    for (size_t i = 0; i < rank(); ++i) {
      ext[i] = dst.extents().extent(i);
    }
    size_t offset = 0;

    auto index_lt = [](const md_index &a, const md_index &b) -> bool {
      for (size_t i = 0; i < rank(); ++i) {
        if (!(a[i] < b[i])) {
          return false;
        }
      }
      return true;
    };

    auto increment = [&]() -> void {
      for (size_t i = 0; i < rank(); ++i) {
        ++index[i];
        if (index[i] >= ext[i] && i != rank() - 1 /* don't wrap final index*/) {
          index[i] = 0;
        } else {
          break;
        }
      }

      ++offset;
    };

    for (; index_lt(index, ext); increment()) {
      dst[index] = src[offset];
    }
  }

  static constexpr bool is_reference_counted() { return true; }
};

#endif  // KOKKOSCOMM_ENABLE_MDSPAN
}  // namespace KokkosComm
