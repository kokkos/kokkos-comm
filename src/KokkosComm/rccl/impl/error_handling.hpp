// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstdio>
#include <string_view>

#include <Kokkos_Core.hpp>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#define KC_HIP_CHECK(expr)                                                                            \
  ([&]() {                                                                                            \
    hipError_t kcErr = (expr);                                                                        \
    if (hipSuccess != kcErr) {                                                                        \
      std::fprintf(stderr, "%s:%d: error (HIP): %s\n", __FILE__, __LINE__, hipGetErrorString(kcErr)); \
    }                                                                                                 \
  }())

#define KC_RCCL_CHECK(expr)                                                                             \
  ([&]() {                                                                                              \
    ncclResult_t kcRes = (expr);                                                                        \
    if (ncclSuccess != kcRes) {                                                                         \
      std::fprintf(stderr, "%s:%d: error (RCCL): %s\n", __FILE__, __LINE__, ncclGetErrorString(kcRes)); \
    }                                                                                                   \
  }())

namespace KokkosComm::rccl {

inline auto fail_if(bool condition, std::string_view error_msg) -> void {
  if (condition) {
    std::fprintf(
        stderr, "error: Kokkos Comm (RCCL) failed with `%.*s`\n", static_cast<int>(error_msg.size()), error_msg.data()
    );
    Kokkos::abort(error_msg.data());
  }
}

inline auto fail_if(bool condition, std::string_view error_msg, ncclComm_t comm) -> void {
  if (condition) {
#ifdef KOKKOSCOMM_ABORT_ON_ERROR
    std::fprintf(
        stderr, "error: Kokkos Comm (RCCL) failed with `%.*s`\n", static_cast<int>(error_msg.size()), error_msg.data()
    );
    ncclCommAbort(comm);
#else
    Kokkos::abort(error_msg.data());
#endif
  }
}

}  // namespace KokkosComm::rccl
