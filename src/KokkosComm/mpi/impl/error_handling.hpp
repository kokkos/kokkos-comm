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

}  // namespace KokkosComm::mpi

#ifdef KOKKOSCOMM_ABORT_ON_ERROR
#define KC_MPI_ON_ERROR_IMPL_(make_ret, err) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE)
#else
#define KC_MPI_ON_ERROR_IMPL_(make_ret, err) return make_ret(err)
#endif

#define KC_ERR_TO_UNEXPECTED_(e) tl::unexpected(e)
#define KC_ERR_TO_REQUEST_(e) ::KokkosComm::Request<::KokkosComm::MpiSpace>::failed(e)

#define KC_MPI_FAIL_IF_IMPL(cond, code, make_ret)                                                                \
  do {                                                                                                           \
    if (cond) {                                                                                                  \
      std::cerr << "Error: KokkosComm check `" #cond "` failed at " << __FILE__ << ":" << __LINE__ << std::endl; \
      KC_MPI_ON_ERROR_IMPL_(make_ret, (::KokkosComm::Error{code}));                                              \
    }                                                                                                            \
  } while (0)

#define KC_MPI_CHECK_IMPL(call, make_ret)                                                                         \
  do {                                                                                                            \
    int mpi_err_ = (call);                                                                                        \
    if (mpi_err_ != MPI_SUCCESS) {                                                                                \
      char mpi_msg_[MPI_MAX_ERROR_STRING];                                                                        \
      int mpi_len_ = 0;                                                                                           \
      if (MPI_Error_string(mpi_err_, mpi_msg_, &mpi_len_) != MPI_SUCCESS) mpi_len_ = 0;                           \
      std::cerr << "Error: MPI call `" #call "` failed at " << __FILE__ << ":" << __LINE__ << " with error code " \
                << mpi_err_ << " (" << std::string_view(mpi_msg_, mpi_len_) << ")" << std::endl;                  \
      return make_ret((::KokkosComm::Error{::KokkosComm::ErrorCode::BackendError, mpi_err_}));                    \
    }                                                                                                             \
  } while (0)

/// For functions returning status_type
#define KC_MPI_FAIL_IF(cond, code) KC_MPI_FAIL_IF_IMPL(cond, code, KC_ERR_TO_UNEXPECTED_)
#define KC_MPI_CHECK(call) KC_MPI_CHECK_IMPL(call, KC_ERR_TO_UNEXPECTED_)

/// For functions returning Request<MpiSpace>
#define KC_MPI_FAIL_IF_REQ(cond, code) KC_MPI_FAIL_IF_IMPL(cond, code, KC_ERR_TO_REQUEST_)
#define KC_MPI_CHECK_REQ(call) KC_MPI_CHECK_IMPL(call, KC_ERR_TO_REQUEST_)
