// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>

namespace KokkosComm::Impl {

template <KokkosView V>
inline constexpr bool needs_staging_v =
    !Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename V::memory_space>::accessible;

/// Stage view on the host for non-GPU-aware communications.
/// No-op if `view` is device-accessible from the host.
template <KokkosView V>
auto stage_for(const V& view) {
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
}

/// Copy back to device (e.g. for receive operations).
/// No-op if `dst` is device-accessible from the host.
template <KokkosExecutionSpace E, KokkosView Dst, KokkosView Src>
auto copy_back(const E& space, Dst& dst, const Src& src) -> void {
  if constexpr (needs_staging_v<Dst>) {
    Kokkos::deep_copy(space, dst, src);
  }
}

}  // namespace KokkosComm::Impl
