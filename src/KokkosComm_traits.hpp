// Copyright 2023 Carl Pearson
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "KokkosComm_mdspan.hpp"

namespace KokkosComm {

template <typename T> concept KokkosView = Kokkos::is_view_v<T>;

#if KOKKOSCOMM_ENABLE_MDSPAN
template <typename T> concept Mdspan = is_mdspan_v<T>;
#endif

template <typename T> struct Traits {
  static_assert(std::is_void_v<T>,
                "KokkosComm::Traits not specialized for type");
};

/*! \brief This can be specialized to do custom behavior for a particular view*/
template <KokkosView View> struct Traits<View> {
  static bool is_contiguous(const View &v) { return v.span_is_contiguous(); }
  static bool needs_unpack(const View &v) { return !is_contiguous(v); }
  static bool needs_pack(const View &v) { return !is_contiguous(v); }

  static auto data_handle(const View &v) { return v.data(); }

  using non_const_packed_view_type =
      Kokkos::View<typename View::non_const_data_type,
                   typename View::array_layout, typename View::memory_space>;
  using packed_view_type =
      Kokkos::View<typename View::data_type, typename View::array_layout,
                   typename View::memory_space>;

  static size_t span(const View &v) { return v.span(); }

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

  static constexpr bool is_reference_counted() { return true; }

  static constexpr size_t rank() { return View::rank; }
};
#if KOKKOSCOMM_ENABLE_MDSPAN

template <Mdspan Span> struct Traits<Span> {
  static bool is_contiguous(const Span &v) { return true; }
  static bool needs_unpack(const Span &v) { return !is_contiguous(v); }
  static bool needs_pack(const Span &v) { return !is_contiguous(v); }

  static auto data_handle(const Span &v) { return v.data_handle(); }

  using non_const_packed_view_type = std::vector<typename Span::value_type>;
  using packed_view_type = std::vector<typename Span::value_type>;

  static auto data_handle(non_const_packed_view_type &v) { return v.data(); }

  static size_t span(const packed_view_type &packed) { return packed.size(); }
  static size_t span(const Span &v) {
    throw std::runtime_error("unimplemented");
  }

  template <typename ExecSpace>
  static void pack(const ExecSpace &space,
                   const non_const_packed_view_type &dst, const Span &src) {
    throw std::runtime_error("unimplemented");
  }

  template <typename ExecSpace>
  static void unpack(const ExecSpace &space, Span &dst,
                     const non_const_packed_view_type &src) {
    throw std::runtime_error("unimplemented");
  }

  static constexpr bool is_reference_counted() { return true; }

  static constexpr size_t rank() { return Span::extents_type::rank(); }
};

#endif // KOKKOSCOMM_ENABLE_MDSPAN
} // namespace KokkosComm
