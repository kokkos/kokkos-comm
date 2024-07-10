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

#include <string>

#include <Kokkos_Core.hpp>

#include "KokkosComm_concepts.hpp"
#include "KokkosComm_traits.hpp"

namespace KokkosComm::Impl {

template <KokkosView View>
struct contiguous_view {
  using type = Kokkos::View<typename View::non_const_data_type, Kokkos::LayoutRight, typename View::memory_space>;
};

template <KokkosView View>
using contiguous_view_t = contiguous_view<View>::type;

template <KokkosView View, KokkosExecutionSpace Space>
auto allocate_contiguous_for(const Space &space, const std::string &label, View &v) {
  using non_const_packed_view_type = contiguous_view_t<View>;

  if constexpr (KokkosComm::rank<View>() == 1) {
    return non_const_packed_view_type(Kokkos::view_alloc(space, Kokkos::WithoutInitializing, label), v.extent(0));
  } else if constexpr (KokkosComm::rank<View>() == 2) {
    return non_const_packed_view_type(Kokkos::view_alloc(space, Kokkos::WithoutInitializing, label), v.extent(0),
                                      v.extent(1));
  } else {
    static_assert(std::is_void_v<View>, "allocate_contiguous_for for views > rank 2 not implemented");
  }
}

template <KokkosExecutionSpace Space, KokkosView DstView, KokkosView SrcView>
auto resize_contiguous_for(const Space &space, DstView &out, const SrcView &in) {
  static_assert(DstView::rank == SrcView::rank, "");

  if constexpr (KokkosComm::rank<DstView>() == 1) {
    Kokkos::realloc(Kokkos::view_alloc(space, Kokkos::WithoutInitializing), out, in.extent(0));
  } else if constexpr (KokkosComm::rank<DstView>() == 2) {
    Kokkos::realloc(Kokkos::view_alloc(space, Kokkos::WithoutInitializing), out, in.extent(0), in.extent(1));
  } else {
    static_assert(std::is_void_v<DstView>, "realloc_contiguous_for for views > rank 2 not implemented");
  }
}

}  // namespace KokkosComm::Impl