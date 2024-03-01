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