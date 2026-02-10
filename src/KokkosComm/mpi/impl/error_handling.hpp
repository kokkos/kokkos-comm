// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstdio>
#include <string_view>

#include <mpi.h>
#include <Kokkos_Core.hpp>

namespace KokkosComm::mpi {

inline auto fail_if(bool condition, std::string_view error_msg, MPI_Comm comm = MPI_COMM_WORLD) -> void {
  if (condition) {
#ifdef KOKKOSCOMM_ABORT_ON_ERROR
    std::fprintf(stderr, "error: Kokkos Comm(MPI) failed with `%.*s`\n", static_cast<int>(error_msg.size()),
                 error_msg.data());
    MPI_Abort(comm, EXIT_FAILURE);
#else
    Kokkos::abort(error_msg.data());
#endif
  }
}

}  // namespace KokkosComm::mpi
