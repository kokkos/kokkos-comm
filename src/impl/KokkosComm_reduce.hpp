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

#include "KokkosComm_traits.hpp"

// impl
#include "KokkosComm_allocate.hpp"
#include "KokkosComm_pack.hpp"
#include "KokkosComm_types.hpp"
#include "KokkosComm_unpack.hpp"

namespace KokkosComm::Impl {
template <typename SendView, typename RecvView, typename ExecSpace>
void reduce(const ExecSpace &space, const SendView &sv, const RecvView &rv,
            MPI_Op op, int root, MPI_Comm comm) {

  const int rank = [=]() -> int {
    int _r;
    MPI_Comm_rank(comm, &_r);
    return _r;
  }();

  if (KokkosComm::Traits<SendView>::needs_pack(sv)) {

    // for reduce, whatever the packed scalar is needs to be convertible to an
    // MPI type, so the MPI implementation can operate on it
    using PackedView =
        typename KokkosComm::Traits<SendView>::non_const_packed_view_type;
    using PackedScalar = typename PackedView::non_const_value_type;

    auto packed_sv = allocate_packed_for(space, "packed_sv", sv);
    pack(space, packed_sv, sv);
    if ((root == rank) && KokkosComm::Traits<RecvView>::needs_unpack(rv)) {
      auto packed_rv = allocate_packed_for(space, "packed_rv", rv);
      space.fence();
      MPI_Reduce(packed_sv.data(), packed_rv.data(), packed_sv.span(),
                 mpi_type_v<PackedScalar>, op, root, comm);
      unpack(space, rv, packed_rv);
    } else {
      space.fence();
      MPI_Reduce(packed_sv.data(), rv.data(), packed_sv.span(),
                 mpi_type_v<PackedScalar>, op, root, comm);
    }
  } else {
    using SendScalar = typename SendView::value_type;
    if ((root == rank) && KokkosComm::Traits<RecvView>::needs_unpack(rv)) {
      auto packed_rv = allocate_packed_for(space, "packed_rv", rv);
      space.fence();
      MPI_Reduce(sv.data(), packed_rv.data(), sv.span(), mpi_type_v<SendScalar>,
                 op, root, comm);
      unpack(space, rv, packed_rv);
    } else {
      space.fence();
      MPI_Reduce(sv.data(), rv.data(), sv.span(), mpi_type_v<SendScalar>, op,
                 root, comm);
    }
  }
  space.fence();
}
} // namespace KokkosComm::Impl