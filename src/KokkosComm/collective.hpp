// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <utility>

#include <Kokkos_Core.hpp>

#include "fwd.hpp"
#include "concepts.hpp"

namespace KokkosComm {

template <KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace,
          CommunicationSpace CommSpace   = DefaultCommunicationSpace>
void barrier(Handle<ExecSpace, CommSpace> &&h) {
  Impl::Barrier<ExecSpace, CommSpace>{std::forward<Handle<ExecSpace, CommSpace>>(h)};
}

}  // namespace KokkosComm
