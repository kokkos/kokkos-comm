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

#include <Kokkos_Core.hpp>

// src
#include "KokkosComm_traits.hpp"

template <typename Dst, typename Src, typename ExecSpace>
void unpack(const ExecSpace &space, const Dst &dst, const Src &src) {
  Kokkos::Tools::pushRegion("KokkosComm::unpack");
  KokkosComm::Traits<Src>::unpack(space, dst, src);
  Kokkos::Tools::popRegion();
}