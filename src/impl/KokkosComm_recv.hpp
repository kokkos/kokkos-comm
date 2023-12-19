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

// impl
#include "KokkosComm_unpack.hpp"

namespace KokkosComm::Impl {
template <typename RecvView, typename ExecSpace>
void recv(const ExecSpace &space, const RecvView &rv, int src, int tag,
          MPI_Comm comm) {
  if (KokkosComm::Traits<RecvView>::needs_unpack(rv)) {
    using PackedScalar =
        typename KokkosComm::Traits<RecvView>::packed_view_type::value_type;
    auto packed = allocate_packed_for(space, "packed", rv);
    space.fence();
    MPI_Recv(packed.data(), packed.span() * sizeof(PackedScalar), MPI_PACKED,
             src, tag, comm, MPI_STATUS_IGNORE);
    unpack(space, rv, packed);
    space.fence();
  } else {
    using RecvScalar = typename RecvView::value_type;
    MPI_Recv(rv.data(), rv.span(), mpi_type_v<RecvScalar>, src, tag, comm,
             MPI_STATUS_IGNORE);
  }
}
} // namespace KokkosComm::Impl