// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include "concepts.hpp"

namespace KokkosComm {

template <typename T>
struct Traits {
  static_assert(std::is_void_v<T>, "KokkosComm::Traits not specialized for type");
};

/*! \brief This can be specialized to do custom behavior for a particular view*/
template <KokkosView V>
struct Traits<V> {
  using non_const_packed_view_type = Kokkos::View<typename V::non_const_data_type,
                                                  typename V::execution_space::array_layout, typename V::memory_space>;
  using packed_view_type =
      Kokkos::View<typename V::data_type, typename V::execution_space::array_layout, typename V::memory_space>;
};

/// @returns A pointer to the underlying data.
template <KokkosView V>
[[nodiscard]] constexpr auto data_handle(const V& view) -> V::pointer_type {
  return view.data();
}

/// @returns The span between the elements with the lowest and highest address.
template <KokkosView V>
[[nodiscard]] constexpr auto span(const V& view) -> V::size_type {
  return view.span();
}

/// @returns The rank of the View.
template <KokkosView V>
[[nodiscard]] constexpr auto rank() -> V::size_type {
  return V::rank;
}
template <KokkosView V>
[[nodiscard]] constexpr auto rank([[maybe_unused]] const V& view) -> V::size_type {
  return rank<V>();
}

/// @returns The number of elements in extent `i`.
template <KokkosView V>
[[nodiscard]] constexpr auto extent(const V& view, int i) -> V::size_type {
  return view.extent(i);
}

/// @returns The stride of elements on extent `i`.
template <KokkosView V>
[[nodiscard]] constexpr auto stride(const V& view, int i) -> V::size_type {
  return view.stride(i);
}

/// @returns Always true for Kokkos Views.
template <KokkosView V>
[[nodiscard]] constexpr auto is_reference_counted() -> bool {
  return true;
}

/// @returns True if, and only if, the product of extents is equal to the span.
template <KokkosView V>
[[nodiscard]] auto is_contiguous(const V& view) -> bool {
  return view.span_is_contiguous();
}

}  // namespace KokkosComm
