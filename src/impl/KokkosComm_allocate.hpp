#pragma once

#include "KokkosComm_traits.hpp"

namespace KokkosComm::Impl {

template <typename ExecSpace, typename View>
typename KokkosComm::Traits<View>::non_const_packed_view_type
allocate_packed_for(const ExecSpace &space, const std::string &label,
                    const View &v) {

  if constexpr (View::rank == 1) {
    return typename KokkosComm::Traits<View>::non_const_packed_view_type(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing, label),
        v.extent(0));
  } else {
    static_assert(std::is_void_v<View>,
                  "allocate_packed only supports rank-1 views");
  }
}

} // namespace KokkosComm::Impl