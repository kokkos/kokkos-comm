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

#include "impl/KokkosComm_reduce.hpp"

#include <Kokkos_Core.hpp>

namespace KokkosComm {

template <typename SendView, typename RecvView, typename ExecSpace>
void reduce(const ExecSpace &space, const SendView &sv, const RecvView &rv,
            MPI_Op op, int root, MPI_Comm comm) {
  return Impl::reduce(space, sv, rv, op, root, comm);
}

} // namespace KokkosComm
