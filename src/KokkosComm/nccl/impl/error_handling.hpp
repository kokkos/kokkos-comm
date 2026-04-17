// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstdio>
#include <string_view>

#include <nccl.h>
#include <Kokkos_Core.hpp>

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
