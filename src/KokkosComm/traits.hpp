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
template <KokkosView View>
struct Traits<View> {
  using non_const_packed_view_type =
      Kokkos::View<typename View::non_const_data_type, typename View::array_layout, typename View::memory_space>;
  using packed_view_type =
      Kokkos::View<typename View::data_type, typename View::array_layout, typename View::memory_space>;
};

template <KokkosView View>
auto data_handle(const View &v) {
  return v.data();
}

// return span in elements between the elements with the lowest and highest address
template <KokkosView View>
auto span(const View &v) {
  return v.span();
}

// true iff product of extents is span
template <KokkosView View>
bool is_contiguous(const View &v) {
  return v.span_is_contiguous();
}

template <KokkosView View>
constexpr size_t rank() {
  return View::rank;
}

template <KokkosView View>
size_t extent(const View &v, const int i) {
  return v.extent(i);
}
template <KokkosView View>
size_t stride(const View &v, const int i) {
  return v.stride(i);
}

template <KokkosView View>
constexpr bool is_reference_counted() {
  return true;
}

}  // namespace KokkosComm
