// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstdio>
#include <iostream>
#include <string_view>

#include <nccl.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/error.hpp>

namespace KokkosComm::nccl {

inline auto print_cuda_error_hint() -> void {
  cudaError_t cuda_hint = cudaPeekAtLastError();
  std::cerr << "  CUDA last error (hint, may be unrelated): " << cudaGetErrorName(cuda_hint) << " ("
            << cudaGetErrorString(cuda_hint) << ");" << std::endl;
}

/// @brief Prints a diagnostic for a failed NCCL call, plus a CUDA hint if NCCL reports an unhandled CUDA error.
/// @param err The error code returned by NCCL.
/// @param call The text of the failed call.
inline auto print_nccl_error(ncclResult_t err, std::string_view call, const char* file, int line) -> void {
  std::cerr << "Error: NCCL call `" << call << "` failed at " << file << ":" << line << " with error code "
            << static_cast<int>(err) << " (" << ncclGetErrorString(err) << ")" << std::endl;
  if (err == ncclUnhandledCudaError) {
    print_cuda_error_hint();
  }
}

/// @brief Prints a diagnostic for a failed CUDA call.
/// @param err The error code returned by CUDA.
/// @param call The text of the failed call.
inline auto print_cuda_error(cudaError_t err, std::string_view call, const char* file, int line) -> void {
  std::cerr << "Error: CUDA call `" << call << "` failed at " << file << ":" << line << " with error code "
            << static_cast<int>(err) << " (" << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << ")"
            << std::endl;
}

}  // namespace KokkosComm::nccl

namespace KokkosComm::nccl::deprecated {

/// @deprecated Aborts on failure. Only kept for `test`, `wait_all` and `wait_any`, which do not report errors yet.
/// New code must use `KC_NCCL_FAIL_IF_REQ` / `KC_NCCL_CHECK_REQ` / `KC_CUDA_CHECK_REQ`.
inline auto fail_if(bool condition, std::string_view error_msg) -> void {
  if (condition) {
    std::fprintf(
        stderr, "error: Kokkos Comm (NCCL) failed with `%.*s`\n", static_cast<int>(error_msg.size()), error_msg.data()
    );
    Kokkos::abort(error_msg.data());
  }
}

/// @deprecated See the overload above.
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

}  // namespace KokkosComm::nccl::deprecated

// Error-checking macros for functions returning `Request<NcclSpace>`.
// On error, they print a diagnostic and either return a failed request, or abort if `KOKKOSCOMM_ABORT_ON_ERROR` is set.

#ifdef KOKKOSCOMM_ABORT_ON_ERROR
#define KC_NCCL_ON_ERROR_IMPL_(err) Kokkos::abort("KokkosComm (NCCL): aborting on error")
#else
#define KC_NCCL_ON_ERROR_IMPL_(err) return ::KokkosComm::Request<::KokkosComm::Experimental::NcclSpace>::failed(err)
#endif

/// Fails the enclosing function with `code` if `cond` holds.
#define KC_NCCL_FAIL_IF_REQ(cond, code)                                  \
  do {                                                                   \
    if (cond) {                                                          \
      ::KokkosComm::Impl::print_check_failed(#cond, __FILE__, __LINE__); \
      KC_NCCL_ON_ERROR_IMPL_((::KokkosComm::Error{code}));               \
    }                                                                    \
  } while (0)

/// Fails the enclosing function with `ErrorCode::NcclError` if `call` does not return `ncclSuccess`.
#define KC_NCCL_CHECK_REQ(call)                                                                                       \
  do {                                                                                                                \
    ncclResult_t nccl_err_ = (call);                                                                                  \
    if (nccl_err_ != ncclSuccess) {                                                                                   \
      ::KokkosComm::nccl::print_nccl_error(nccl_err_, #call, __FILE__, __LINE__);                                     \
      KC_NCCL_ON_ERROR_IMPL_((::KokkosComm::Error{::KokkosComm::ErrorCode::NcclError, static_cast<int>(nccl_err_)})); \
    }                                                                                                                 \
  } while (0)

/// Fails the enclosing function with `ErrorCode::CudaError` if `call` does not return `cudaSuccess`.
#define KC_CUDA_CHECK_REQ(call)                                                                                       \
  do {                                                                                                                \
    cudaError_t cuda_err_ = (call);                                                                                   \
    if (cuda_err_ != cudaSuccess) {                                                                                   \
      ::KokkosComm::nccl::print_cuda_error(cuda_err_, #call, __FILE__, __LINE__);                                     \
      KC_NCCL_ON_ERROR_IMPL_((::KokkosComm::Error{::KokkosComm::ErrorCode::CudaError, static_cast<int>(cuda_err_)})); \
    }                                                                                                                 \
  } while (0)
