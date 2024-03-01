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

#include "KokkosComm_collective.hpp"
#include "KokkosComm_version.hpp"
#include "impl/KokkosComm_isend.hpp"
#include "impl/KokkosComm_recv.hpp"
#include "impl/KokkosComm_send.hpp"

#include <Kokkos_Core.hpp>

namespace KokkosComm {

template <typename SendView, typename ExecSpace>
Req isend(const ExecSpace &space, const SendView &sv, int dest, int tag,
          MPI_Comm comm) {
  return Impl::isend(space, sv, dest, tag, comm);
}

template <typename SendView, typename ExecSpace>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag,
          MPI_Comm comm) {
  return Impl::send(space, sv, dest, tag, comm);
}

template <typename Recv, typename ExecSpace>
void recv(const ExecSpace &space, Recv &sv, int src, int tag, MPI_Comm comm) {
  return Impl::recv(space, sv, src, tag, comm);
}

} // namespace KokkosComm
