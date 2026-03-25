// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>

namespace KokkosComm::Impl {

template <KokkosView View>
struct contiguous_view {
  using type = Kokkos::View<typename View::non_const_data_type, typename View::execution_space::array_layout,
                            typename View::memory_space>;
};

template <KokkosView View>
using contiguous_view_t = contiguous_view<View>::type;

/// @brief Allocate a contiguous View suitable for packing a non-contiguous View.
/// @tparam Exec A Kokkos Execution Space type.
/// @tparam View A Kokkos View type.
/// @param exec The execution space instance in which to perform the view allocation.
/// @param v The View to make a suitable contiguous allocation for.
/// @param label The label to give to the allocated contiguous View. Defaults to "contiguous_view".
template <KokkosExecutionSpace Exec, KokkosView View>
auto allocate_contiguous_for(const Exec& exec, const View& v, const std::string& label = "contiguous_view") {
  using ContigView = contiguous_view_t<View>;
  return [&]<size_t... Is>(std::index_sequence<Is...>) {
    return ContigView(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, label), v.extent(Is)...);
  }
  (std::make_index_sequence<rank<View>()>{});
}

/// @brief Resize a View into a contiguous one suitable for packing a non-contiguous View.
/// @tparam Exec A Kokkos Execution Space type.
/// @tparam DstV The Kokkos View type of the destination View to resize.
/// @tparam DstV The Kokkos View type of the source View to resize for.
/// @param exec The execution space instance in which to perform the view reallocation.
/// @param dst The View to resize.
/// @param src The View to make a suitable contiguous resize for.
template <KokkosExecutionSpace Exec, KokkosView DstV, KokkosView SrcV>
auto resize_contiguous_for(const Exec& exec, const DstV& dst, const SrcV& src) {
  static_assert(rank<DstV>() == rank<SrcV>(),
                "KokkosComm::Impl::resize_contiguous_for: source and destination Views must have the same rank");
  [&]<size_t... Is>(std::index_sequence<Is...>) {
    Kokkos::realloc(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing), dst, src.extent(Is)...);
  }
  (std::make_index_sequence<rank<DstV>()>{});
}

}  // namespace KokkosComm::Impl
