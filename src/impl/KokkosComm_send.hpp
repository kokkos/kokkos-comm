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

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include "KokkosComm_pack.hpp"

// impl
#include "KokkosComm_pack.hpp"

namespace KokkosComm::Impl {
template <typename SendView, typename ExecSpace>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag,
          MPI_Comm comm) {
  if (KokkosComm::Traits<SendView>::needs_pack(sv)) {
    if constexpr (SendView::rank == 1) {
      using SendScalar = typename KokkosComm::Traits<
          SendView>::non_const_packed_view_type::value_type;
      auto packed = allocate_packed_for(space, "packed", sv);
      pack(space, packed, sv);
      space.fence();
      MPI_Send(packed.data(), packed.span() * sizeof(SendScalar), MPI_PACKED,
               dest, tag, comm);
    } else {
      static_assert(std::is_void_v<SendView>,
                    "send only supports rank-1 views");
    }
  } else {
    using SendScalar = typename SendView::value_type;
    MPI_Send(sv.data(), sv.span(), mpi_type_v<SendScalar>, dest, tag, comm);
  }
}
} // namespace KokkosComm::Impl