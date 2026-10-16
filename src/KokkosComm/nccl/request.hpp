// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/fwd.hpp>
#include "nccl_space.hpp"

#include "impl/error_handling.hpp"

namespace KokkosComm {

/// @brief Request specialization for the NCCL communication space.
template <>
class Request<Experimental::NcclSpace> {
 public:
  using communication_space = Experimental::NcclSpace;
  using request_type        = Experimental::NcclSpace::request_type;
  using rank_type           = Experimental::NcclSpace::rank_type;

  /// @brief Constructs a `Request`.
  explicit Request() : request_(nullptr) {}

  /// @brief Capture the state of a `cudaStream_t` for request encapsulation.
  /// @param stream The stream to capture for request encapsulation.
  auto capture_stream_state(cudaStream_t stream) noexcept -> void {
    if (request_ != nullptr) {
      KC_CUDA_CHECK(cudaEventDestroy(request_));
    }
    KC_CUDA_CHECK(cudaEventCreate(&request_, cudaEventDisableTiming));
    KC_CUDA_CHECK(cudaEventRecord(request_, stream));
  }

  /// @brief Destructor.
  ~Request() noexcept {
    if (request_ != nullptr) {
      KC_CUDA_CHECK(cudaEventDestroy(request_));
    }
  };

  /// @brief Copy constructor is deleted because a `Request` can only be moved.
  Request(const Request&) = delete;
  /// @brief Copy assignment operator is deleted because a `Request` can only be moved.
  auto operator=(const Request&) -> Request& = delete;
  /// @brief Move constructor.
  Request(Request&& other) noexcept
      : request_(std::exchange(other.request_, nullptr)), callbacks_(std::move(other.callbacks_)) {}
  /// @brief Move assignment operator.
  auto operator=(Request&& other) noexcept -> Request& {
    if (this != &other) {
      if (request_ != nullptr) {
        KC_CUDA_CHECK(cudaEventDestroy(request_));
      }
      request_   = std::exchange(other.request_, nullptr);
      callbacks_ = std::move(other.callbacks_);
    }
    return *this;
  }

  /// @return A reference to the underlying `cudaEvent_t` object.
  [[nodiscard]] constexpr auto request() noexcept -> request_type& { return request_; }
  /// @return A const reference to the underlying `cudaEvent_t` object.
  [[nodiscard]] constexpr auto request() const noexcept -> const request_type& { return request_; }
  /// @return A pointer to the underlying `cudaEvent_t` object.
  [[nodiscard]] constexpr auto request_ptr() noexcept -> request_type* { return &request_; }
  /// @return A const pointer to the underlying `cudaEvent_t` object.
  [[nodiscard]] constexpr auto request_ptr() const noexcept -> const request_type* { return &request_; }

  /// @brief Adds a function to a list of callbacks to be invoked after the request's completion.
  /// @param cb The callback function to register.
  auto add_callback(std::function<void()>&& cb) -> void { callbacks_.push_back(cb); }

  /// @brief Captures a Kokkos View to extend its lifetime until the request's completion.
  /// @tparam V A Kokkos View type.
  /// @param view The Kokkos View to capture for lifetime extension.
  template <KokkosView V>
  auto extend_view_lifetime(const V& view) -> void {
    // Unmanaged views don't own the underlying buffer, so no need to extend their lifetime
    if (view.use_count() != 0) {
      add_callback([view]() {});
    }
  }

  /// @brief Waits on the request until completion of the associated operation.
  auto wait() -> void {
    cudaError_t err = cudaEventSynchronize(request_);
    // FIXME: Do something smarter with `err` for better error reporting
    nccl::fail_if(err != cudaSuccess, "KokkosComm::Request::wait: request completion failed");

    execute_all_callbacks();
  }

  /// @brief Queries the request for the completion of the associated operation.
  /// If the operation has completed, all callbacks are executed upon return, similarly to having called `wait`.
  /// @return True if the request has completed or is null/inactive, false otherwise.
  [[nodiscard]] auto test() -> bool {
    cudaError_t err = cudaEventQuery(request_);
    if (err == cudaSuccess) {
      execute_all_callbacks();
      return true;
    } else if (err == cudaErrorNotReady) {
      return false;
    }

    // FIXME: Do something smarter with `err` for better error reporting
    nccl::fail_if(err != cudaSuccess, "KokkosComm::Request::wait: request completion failed");
    // unreachable
    return false;
  }

 private:
  request_type request_;
  std::vector<std::function<void()>> callbacks_;

  /// @brief Executes all the callbacks registered on the request.
  auto execute_all_callbacks() -> void {
    for (auto& cb : callbacks_) {
      cb();
    }
    callbacks_.clear();
  }

  friend auto wait(Request<communication_space>& request) -> void;
  friend auto wait(Request<communication_space>&& request) -> void;
  friend auto wait_all(std::span<Request<communication_space>> requests) -> void;
  friend auto wait_any(std::span<Request<communication_space>> requests) -> std::optional<rank_type>;
  friend auto test(Request<communication_space>& request) -> bool;
};

/// @brief Waits on the request until completion of the associated operation.
/// @param request A reference on the request to wait for completion.
inline auto wait(Request<Experimental::NcclSpace>& request) -> void { request.wait(); }
/// @brief Waits on the request until completion of the associated operation.
/// @param request An r-value reference on the request, consumed upon completion.
inline auto wait(Request<Experimental::NcclSpace>&& request) -> void { request.wait(); }

/// @brief Waits for completion of all passed requests.
/// @param requests The list of requests to complete.
inline auto wait_all(std::span<Request<Experimental::NcclSpace>> requests) -> void {
  if (requests.empty()) {
    return;
  }

  int remaining = requests.size();
  // Poll until all requests are completed
  //
  // NOTE: While this is an active-wait loop, it should be the best compromise for performance.
  // Other implementation strategies could be:
  // - Complete requests in parallel by spawning threads
  // - Complete requests one at a time in a sequential loop
  while (remaining > 0) {
    for (auto& req : requests) {
      cudaError_t err = cudaEventQuery(req.request());
      if (err == cudaSuccess) {
        req.execute_all_callbacks();
        remaining--;
      } else if (err == cudaErrorNotReady) {
        continue;
      } else {
        // FIXME: Do something smarter with `err` for better error reporting
        nccl::fail_if(err != cudaSuccess, "KokkosComm::Request::wait_all: request completions failed");
      }
    }
  }
}

/// @brief Waits for the completion of one request among all passed requests.
/// @param requests The list of requests to try to complete.
/// @return The index of the request within the passed list upon successful completion, `std::nullopt` otherwise.
inline auto wait_any(std::span<Request<Experimental::NcclSpace>> requests)
    -> std::optional<typename Request<Experimental::NcclSpace>::rank_type> {
  if (requests.empty()) {
    return std::nullopt;
  }

  // Poll until at least one request is completed
  //
  // NOTE: While this is an active-wait loop, it should be the best compromise for simplicity/performance.
  // Another implementation strategy could be to complete requests in parallel by spawning threads, but this needs
  // synchronization on the first request completion.
  while (true) {
    for (size_t r = 0; r < requests.size(); ++r) {
      cudaError_t err = cudaEventQuery(requests[r].request());
      if (err == cudaSuccess) {
        requests[r].execute_all_callbacks();
        return static_cast<typename Request<Experimental::NcclSpace>::rank_type>(r);
      } else if (err == cudaErrorNotReady) {
        continue;
      } else {
        // FIXME: Do something smarter with `err` for better error reporting
        nccl::fail_if(err != cudaSuccess, "KokkosComm::Request::wait_any: request completion failed");
      }
    }
  }
}

/// @brief Queries the request for completion of the associated operation.
/// @param request A reference on the request to query its completion.
inline auto test(Request<Experimental::NcclSpace>& request) -> bool { return request.test(); }

}  // namespace KokkosComm
