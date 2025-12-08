// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#define KC_NCCL_CHECK(expr)                                                                      \
  ([&]() {                                                                                       \
    ncclResult_t kcRes = (expr);                                                                 \
    if (ncclSuccess != kcRes) {                                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << ": NCCL Error: " << ncclGetErrorString(kcRes); \
    }                                                                                            \
  }())
#endif
