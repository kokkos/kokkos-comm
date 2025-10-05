// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSCOMM_MPI_IMPL_ERROR_HANDLING_HPP
#define KOKKOSCOMM_MPI_IMPL_ERROR_HANDLING_HPP

#include <iostream>
#include <mpi.h>

namespace KokkosComm::mpi {

inline void fail_if(bool condition, const char* error_msg) {
  if (condition) {
#ifdef KOKKOSCOMM_ABORT_ON_ERROR
    std::cerr << error_msg << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#else
    Kokkos::abort(error_msg);
#endif
  }
}

}  // namespace KokkosComm::mpi

#endif  // KOKKOSCOMM_MPI_IMPL_ERROR_HANDLING_HPP
