#pragma once

#include <Kokkos_Core.hpp>

// src
#include "KokkosComm_traits.hpp"

template <typename Dst, typename Src, typename ExecSpace>
void unpack(const ExecSpace &space, const Dst &dst, const Src &src) {
  Kokkos::Tools::pushRegion("KokkosComm::unpack");
  KokkosComm::Traits<Src>::unpack(space, dst, src);
  Kokkos::Tools::popRegion();
}