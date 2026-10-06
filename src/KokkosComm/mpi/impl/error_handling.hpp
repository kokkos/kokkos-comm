// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstdio>
#include <string_view>

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/error.hpp>

namespace KokkosComm::mpi {

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

// Shared core: runs `call`, logs on failure, then returns `make_ret(mpi_err_)`
#define KC_MPI_CHECK_IMPL(call, caller, make_ret)                                       \
  do {                                                                                  \
    int mpi_err_ = (call);                                                              \
    if (mpi_err_ != MPI_SUCCESS) {                                                      \
      char mpi_msg_[MPI_MAX_ERROR_STRING];                                              \
      int mpi_len_ = 0;                                                                 \
      if (MPI_Error_string(mpi_err_, mpi_msg_, &mpi_len_) != MPI_SUCCESS) mpi_len_ = 0; \
      std::cerr << "Error: " << caller << " returned error code " << mpi_err_ << " ("   \
                << std::string_view(mpi_msg_, mpi_len_) << ")" << std::endl;            \
      return make_ret(mpi_err_);                                                        \
    }                                                                                   \
  } while (0)

#define KC_MPI_ERR_TO_UNEXPECTED_(e) tl::unexpected(::KokkosComm::Error{::KokkosComm::MPIError, (e)})
#define KC_MPI_ERR_TO_REQUEST_(e) \
  ::KokkosComm::Request<::KokkosComm::MpiSpace>::failed(::KokkosComm::Error{::KokkosComm::MPIError, (e)})

/// For functions returning status_type (blocking send/recv, etc.)
#define KC_MPI_CHECK(call, caller) KC_MPI_CHECK_IMPL(call, caller, KC_MPI_ERR_TO_UNEXPECTED_)

/// For functions returning Request<MpiSpace> (isend/irecv)
#define KC_MPI_CHECK_REQ(call, caller) KC_MPI_CHECK_IMPL(call, caller, KC_MPI_ERR_TO_REQUEST_)

}  // namespace KokkosComm::mpi
