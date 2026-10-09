// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/error.hpp>

namespace KokkosComm::mpi::deprecated {

/// @deprecated Aborts on failure. Only kept for the blocking `mpi::` functions, `Channel`, `test`, `wait_all` and
/// `wait_any`, which do not report errors yet. New code must use `KC_MPI_FAIL_IF_REQ` / `KC_MPI_CHECK_REQ`.
inline auto fail_if(bool condition, std::string_view error_msg, MPI_Comm comm = MPI_COMM_WORLD) -> void {
  if (condition) {
#ifdef KOKKOSCOMM_ABORT_ON_ERROR
    std::fprintf(
        stderr, "error: Kokkos Comm(MPI) failed with `%.*s`\n", static_cast<int>(error_msg.size()), error_msg.data()
    );
    MPI_Abort(comm, EXIT_FAILURE);
#else
    Kokkos::abort(error_msg.data());
#endif
  }
}

}  // namespace KokkosComm::mpi::deprecated

// Error-checking macros for functions returning `Request<MpiSpace>`.
// On error, they print a diagnostic and either return a failed request, or abort if `KOKKOSCOMM_ABORT_ON_ERROR` is set.

#ifdef KOKKOSCOMM_ABORT_ON_ERROR
#define KC_MPI_ON_ERROR_IMPL_(err) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE)
#else
#define KC_MPI_ON_ERROR_IMPL_(err) return ::KokkosComm::Request<::KokkosComm::MpiSpace>::failed(err)
#endif

/// Fails the enclosing function with `code` if `cond` holds.
#define KC_MPI_FAIL_IF_REQ(cond, code)                                                                           \
  do {                                                                                                           \
    if (cond) {                                                                                                  \
      std::cerr << "Error: KokkosComm check `" #cond "` failed at " << __FILE__ << ":" << __LINE__ << std::endl; \
      KC_MPI_ON_ERROR_IMPL_((::KokkosComm::Error{code}));                                                        \
    }                                                                                                            \
  } while (0)

/// Fails the enclosing function with `ErrorCode::MpiError` if `call` does not return `MPI_SUCCESS`.
#define KC_MPI_CHECK_REQ(call)                                                                                    \
  do {                                                                                                            \
    int mpi_err_ = (call);                                                                                        \
    if (mpi_err_ != MPI_SUCCESS) {                                                                                \
      char mpi_msg_[MPI_MAX_ERROR_STRING];                                                                        \
      int mpi_len_ = 0;                                                                                           \
      if (MPI_Error_string(mpi_err_, mpi_msg_, &mpi_len_) != MPI_SUCCESS) mpi_len_ = 0;                           \
      std::cerr << "Error: MPI call `" #call "` failed at " << __FILE__ << ":" << __LINE__ << " with error code " \
                << mpi_err_ << " (" << std::string_view(mpi_msg_, mpi_len_) << ")" << std::endl;                  \
      KC_MPI_ON_ERROR_IMPL_((::KokkosComm::Error{::KokkosComm::ErrorCode::MpiError, mpi_err_}));                  \
    }                                                                                                             \
  } while (0)
