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

#include "KokkosComm_mdspan.hpp"
#include "KokkosComm_traits.hpp"

namespace KokkosComm::Impl {

template <typename ExecSpace, KokkosView View>
typename KokkosComm::Traits<View>::non_const_packed_view_type
allocate_packed_for(const ExecSpace &space, const std::string &label,
                    const View &v) {

  using KCT = KokkosComm::Traits<View>;

  if constexpr (KCT::rank() == 1) {
    return typename KCT::non_const_packed_view_type(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing, label),
        v.extent(0));
  } else if constexpr (KCT::rank() == 2) {
    return typename KCT::non_const_packed_view_type(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing, label),
        v.extent(0), v.extent(1));
  } else {
    static_assert(std::is_void_v<View>,
                  "allocate_packed only supports rank-1 views");
  }
}
#if KOKKOSCOMM_ENABLE_MDSPAN
template <typename ExecSpace, Mdspan View>
typename KokkosComm::Traits<View>::non_const_packed_view_type
allocate_packed_for(const ExecSpace &space, const std::string &label,
                    const View &v) {

  using KCT = KokkosComm::Traits<View>;

  if constexpr (KCT::rank() == 1) {
    return typename KCT::non_const_packed_view_type(v.extent(0));
  } else if constexpr (KCT::rank() == 2) {
    return typename KCT::non_const_packed_view_type(v.extent(0) * v.extent(1));
  } else {
    static_assert(std::is_void_v<View>,
                  "allocate_packed only supports rank-1 views");
  }
}
#endif // KOKKOSCOMM_ENABLE_MDSPAN

} // namespace KokkosComm::Impl