#pragma once

#include <Kokkos_Core.hpp>

// src
#include "KokkosComm_traits.hpp"

template <typename Dst, typename Src, typename ExecSpace>
void pack(const ExecSpace &space, const Dst &dst, const Src &src) {
  Kokkos::Tools::pushRegion("KokkosComm::pack");
  KokkosComm::Traits<Src>::pack(space, dst, src);
  Kokkos::Tools::popRegion();
}
