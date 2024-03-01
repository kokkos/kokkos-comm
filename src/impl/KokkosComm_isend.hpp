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

#include <iostream>

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include "KokkosComm_request.hpp"

// impl
#include "KokkosComm_mdspan.hpp"
#include "KokkosComm_pack.hpp"

namespace KokkosComm::Impl {

template <typename Span, typename ExecSpace>
KokkosComm::Req isend(const ExecSpace &space, const Span &ss, int dest, int tag,
                      MPI_Comm comm) {
  KokkosComm::Req req;

  using KCT = KokkosComm::Traits<Span>;

  if (KCT::needs_pack(ss)) {
    using PackedScalar = typename KCT::packed_view_type::value_type;
    auto packed = allocate_packed_for(space, "packed", ss);
    pack(space, packed, ss);
    space.fence();
    MPI_Isend(KCT::data_handle(packed),
              KCT::span(packed) * sizeof(PackedScalar), MPI_PACKED, dest, tag,
              comm, &req.mpi_req());
    req.keep_until_wait(packed);
  } else {
    using SendScalar = typename Span::value_type;
    MPI_Isend(KCT::data_handle(ss), KCT::span(ss), mpi_type_v<SendScalar>, dest,
              tag, comm, &req.mpi_req());
    if (KCT::is_reference_counted()) {
      req.keep_until_wait(ss);
    }
  }
  return req;
}

} // namespace KokkosComm::Impl