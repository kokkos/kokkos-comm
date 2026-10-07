// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstdio>
#include <iostream>
#include <string_view>

#include <nccl.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/error.hpp>

#define KC_CUDA_CHECK(expr)                                                                             \
  ([&]() {                                                                                              \
    cudaError_t kcErr = (expr);                                                                         \
    if (cudaSuccess != kcErr) {                                                                         \
      std::fprintf(stderr, "%s:%d: error (CUDA): %s\n", __FILE__, __LINE__, cudaGetErrorString(kcErr)); \
    }                                                                                                   \
  }())

#define KC_NCCL_CHECK(expr)                                                                             \
  ([&]() {                                                                                              \
    ncclResult_t kcRes = (expr);                                                                        \
    if (ncclSuccess != kcRes) {                                                                         \
      std::fprintf(stderr, "%s:%d: error (NCCL): %s\n", __FILE__, __LINE__, ncclGetErrorString(kcRes)); \
    }                                                                                                   \
  }())

#ifdef KOKKOSCOMM_ABORT_ON_ERROR
#define KC_NCCL_ON_ERROR_IMPL_(make_ret, err) Kokkos::abort("KokkosComm (NCCL): aborting on error")
#else
#define KC_NCCL_ON_ERROR_IMPL_(make_ret, err) return make_ret(err)
#endif

#define KC_NCCL_ERR_TO_REQUEST_(e) ::KokkosComm::Request<::KokkosComm::Experimental::NcclSpace>::failed(e)

#define KC_NCCL_FAIL_IF_IMPL(cond, code, make_ret)                                                               \
  do {                                                                                                           \
    if (cond) {                                                                                                  \
      std::cerr << "Error: KokkosComm check `" #cond "` failed at " << __FILE__ << ":" << __LINE__ << std::endl; \
      KC_NCCL_ON_ERROR_IMPL_(make_ret, (::KokkosComm::Error{code}));                                             \
    }                                                                                                            \
  } while (0)

#define KC_NCCL_CHECK_IMPL(call, make_ret)                                                                         \
  do {                                                                                                             \
    ncclResult_t nccl_err_ = (call);                                                                               \
    if (nccl_err_ != ncclSuccess) {                                                                                \
      std::cerr << "Error: NCCL call `" #call "` failed at " << __FILE__ << ":" << __LINE__ << " with error code " \
                << static_cast<int>(nccl_err_) << " (" << ncclGetErrorString(nccl_err_) << ")" << std::endl;       \
      return make_ret((::KokkosComm::Error{::KokkosComm::ErrorCode::BackendError, static_cast<int>(nccl_err_)}));  \
    }                                                                                                              \
  } while (0)

/// For functions returning Request<NcclSpace>
#define KC_NCCL_FAIL_IF_REQ(cond, code) KC_NCCL_FAIL_IF_IMPL(cond, code, KC_NCCL_ERR_TO_REQUEST_)
#define KC_NCCL_CHECK_REQ(call) KC_NCCL_CHECK_IMPL(call, KC_NCCL_ERR_TO_REQUEST_)

namespace KokkosComm::nccl {

inline auto fail_if(bool condition, std::string_view error_msg) -> void {
  if (condition) {
    std::fprintf(
        stderr, "error: Kokkos Comm (NCCL) failed with `%.*s`\n", static_cast<int>(error_msg.size()), error_msg.data()
    );
    Kokkos::abort(error_msg.data());
  }
}

inline auto fail_if(bool condition, std::string_view error_msg, ncclComm_t comm) -> void {
  if (condition) {
#ifdef KOKKOSCOMM_ABORT_ON_ERROR
    std::fprintf(
        stderr, "error: Kokkos Comm (NCCL) failed with `%.*s`\n", static_cast<int>(error_msg.size()), error_msg.data()
    );
    ncclCommAbort(comm);
#else
    Kokkos::abort(error_msg.data());
#endif
  }
}

}  // namespace KokkosComm::nccl
