// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <concepts>
#include <type_traits>

#include "concepts.hpp"

namespace KokkosComm {

template <typename T>
struct Traits {
  static_assert(std::is_void_v<T>, "KokkosComm::Traits not specialized for type");
};

/// @brief A struct that can be specialized to implement custom behavior for a particular Kokkos view.
template <KokkosView V>
struct Traits<V> {
  using non_const_packed_view_type = Kokkos::
      View<typename V::non_const_data_type, typename V::execution_space::array_layout, typename V::memory_space>;
  using packed_view_type =
      Kokkos::View<typename V::data_type, typename V::execution_space::array_layout, typename V::memory_space>;
};

/// @tparam V A Kokkos View type.
/// @param view The Kokkos View to query.
/// @returns The rank (number of dimensions) of the View.
template <KokkosView V>
[[nodiscard]] constexpr auto rank() noexcept -> size_t {
  return V::rank;
}
template <KokkosView V>
[[nodiscard]] constexpr auto rank([[maybe_unused]] const V& view) noexcept -> size_t {
  return V::rank;
}

/// @tparam V A Kokkos View type.
/// @param view The Kokkos View to query.
/// @returns A pointer to the underlying data allocation.
template <KokkosView V>
[[nodiscard]] constexpr auto data_handle(const V& view) noexcept -> V::pointer_type {
  return view.data();
}

/// @tparam V A Kokkos View type.
/// @param view The Kokkos View to query.
/// @returns The product of extents, i.e., the logical number of elements in the View.
template <KokkosView V>
[[nodiscard]] constexpr auto size(const V& view) noexcept -> size_t {
  return view.size();
}

/// @tparam V A Kokkos View type.
/// @param view The Kokkos View to query.
/// @returns The span between the elements of lowest and highest address.
/// The span may be larger than the product of extents due to padding, and or non-contiguous data layout.
template <KokkosView V>
[[nodiscard]] constexpr auto span(const V& view) noexcept -> size_t {
  return view.span();
}

/// @tparam V A Kokkos View type.
/// @tparam I An integral type.
/// @param view The Kokkos View to query.
/// @param i The index of the dimension. Must be smaller than the rank of the View.
/// @returns The extent (number of elements) of the specified dimension.
template <KokkosView V, std::integral I>
[[nodiscard]] constexpr auto extent(const V& view, I i) noexcept -> size_t {
  return view.extent(i);
}

/// @tparam V A Kokkos View type.
/// @tparam I An integral type.
/// @param view The Kokkos View to query.
/// @param i The index of the dimension. Must be smaller than the rank of the View.
/// @returns The stride (the number of elements the mapping advances upon increment) of the specified dimension.
template <KokkosView V, std::integral I>
[[nodiscard]] constexpr auto stride(const V& view, I i) noexcept -> V::size_type {
  return view.stride(i);
}

/// @tparam V A Kokkos View type.
/// @param view The Kokkos View to query.
/// @returns Always true for Kokkos Views.
template <KokkosView V>
[[nodiscard]] constexpr auto is_reference_counted() noexcept -> bool {
  return true;
}
template <KokkosView V>
[[nodiscard]] constexpr auto is_reference_counted([[maybe_unused]] const V& view) noexcept -> bool {
  return true;
}

/// @tparam V A Kokkos View type.
/// @param view The Kokkos View to query.
/// @returns True if, and only if, the product of extents is equal to the span.
template <KokkosView V>
[[nodiscard]] auto is_contiguous(const V& view) noexcept -> bool {
  return view.span_is_contiguous();
}

}  // namespace KokkosComm
