// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <array>
#include <cstdlib>
#include <string_view>

#include <cuda.h>
#include <mpi.h>
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <nccl.h>
#endif

#include <fmt/core.h>

namespace logging {

enum struct Level {
  FATAL,
  ERROR,
  WARN,
  INFO,
  TRACE,
};

using namespace std::string_view_literals;
constexpr std::array level_txt{"FATAL"sv, "ERROR"sv, "WARNING"sv, "INFO"sv, "TRACE"sv};

}  // namespace logging

#define KC_LOG(lvl, ...)                                                                        \
  fmt::println("[{}] {}:{}: {}", logging::level_txt[static_cast<int>(lvl)], __FILE__, __LINE__, \
               fmt::format(__VA_ARGS__))

#define KC_FATAL(...) (KC_LOG(logging::Level::FATAL, __VA_ARGS__), std::exit(EXIT_FAILURE))

#define KC_ERROR(...) KC_LOG(logging::Level::ERROR, __VA_ARGS__)

#define KC_WARN(...) KC_LOG(logging::Level::WARN, __VA_ARGS__)

#define KC_INFO(...) KC_LOG(logging::Level::INFO, __VA_ARGS__)

#define KC_TRACE(...) KC_LOG(logging::Level::TRACE, __VA_ARGS__)

#define KC_CHECK(expr, ...) ((expr) ? void(0) : KC_FATAL(__VA_ARGS__))

#define KC_CUDA_CHECK(expr)                                                                                      \
  ([&]() {                                                                                                       \
    cudaError_t kc_res_ = (expr);                                                                                \
    return kc_res_ == cudaSuccess ? void(0)                                                                      \
                                  : KC_FATAL("CUDA check failed: `" #expr "`: {}", cudaGetErrorString(kc_res_)); \
  }())

#define KC_MPI_CHECK(expr)                                                                            \
  ([&]() {                                                                                            \
    int kc_res_ = (expr);                                                                             \
    return kc_res_ == MPI_SUCCESS ? void(0) : KC_FATAL("MPI check failed: `" #expr "`: {}", kc_res_); \
  }())

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#define KC_NCCL_CHECK(expr)                                                                                      \
  ([&]() {                                                                                                       \
    ncclResult_t kc_res_ = (expr);                                                                               \
    return kc_res_ == ncclSuccess ? void(0)                                                                      \
                                  : KC_FATAL("NCCL check failed: `" #expr "`: {}", ncclGetErrorString(kc_res_)); \
  }())
#endif
