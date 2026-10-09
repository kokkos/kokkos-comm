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

namespace KokkosComm::mpi::Impl {

/// @brief Prints a diagnostic for a failed MPI call, including MPI's own error string.
/// @param err The error code returned by MPI.
/// @param call The text of the failed call.
inline auto print_mpi_error(int err, std::string_view call, const char* file, int line) -> void {
  char msg[MPI_MAX_ERROR_STRING];
  int len = 0;
  if (MPI_Error_string(err, msg, &len) != MPI_SUCCESS) len = 0;
  std::cerr << "Error: MPI call `" << call << "` failed at " << file << ":" << line << " with error code " << err
            << " (" << std::string_view(msg, len) << ")" << std::endl;
}

}  // namespace KokkosComm::mpi::Impl

// Error-checking macros for functions returning `Request<MpiSpace>`.
// On error, they print a diagnostic and either return a failed request, or abort if `KOKKOSCOMM_ABORT_ON_ERROR` is set.

#ifdef KOKKOSCOMM_ABORT_ON_ERROR
#define KC_MPI_ON_ERROR_IMPL_(err) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE)
#else
#define KC_MPI_ON_ERROR_IMPL_(err) return ::KokkosComm::Request<::KokkosComm::MpiSpace>::failed(err)
#endif

/// Fails the enclosing function with `code` if `cond` holds.
#define KC_MPI_FAIL_IF_REQ(cond, code)                                   \
  do {                                                                   \
    if (cond) {                                                          \
      ::KokkosComm::Impl::print_check_failed(#cond, __FILE__, __LINE__); \
      KC_MPI_ON_ERROR_IMPL_((::KokkosComm::Error{code}));                \
    }                                                                    \
  } while (0)

/// Fails the enclosing function with `ErrorCode::MpiError` if `call` does not return `MPI_SUCCESS`.
#define KC_MPI_CHECK_REQ(call)                                                                   \
  do {                                                                                           \
    int mpi_err_ = (call);                                                                       \
    if (mpi_err_ != MPI_SUCCESS) {                                                               \
      ::KokkosComm::mpi::Impl::print_mpi_error(mpi_err_, #call, __FILE__, __LINE__);             \
      KC_MPI_ON_ERROR_IMPL_((::KokkosComm::Error{::KokkosComm::ErrorCode::MpiError, mpi_err_})); \
    }                                                                                            \
  } while (0)
