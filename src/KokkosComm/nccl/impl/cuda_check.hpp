// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#define KC_CUDA_CHECK(expr)                                                                      \
  ([&]() {                                                                                       \
    cudaError_t kcErr = (expr);                                                                  \
    if (cudaSuccess != kcErr) {                                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CUDA Error: " << cudaGetErrorString(kcErr); \
    }                                                                                            \
  }())
#endif
