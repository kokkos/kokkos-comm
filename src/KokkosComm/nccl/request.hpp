// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <functional>
#include <memory>
#include <span>
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
    KC_CUDA_CHECK(cudaEventCreateWithFlags(&request_, cudaEventDisableTiming));
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
  Request(Request&&) = default;
  /// @brief Move assignment operator.
  auto operator=(Request&&) -> Request& = default;

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

  /// @brief Waits on the request, retires its event, and executes completion work. Empty requests are harmless.
  auto wait() -> void {
    if (request_ == nullptr) {
      execute_all_callbacks();
      return;
    }
    cudaError_t err = cudaEventSynchronize(request_);
    // FIXME: Do something smarter with `err` for better error reporting
    nccl::fail_if(err != cudaSuccess, "KokkosComm::Request::wait: request completion failed");

    retire_event();
    execute_all_callbacks();
  }

  /// @brief Queries the request for the completion of the associated operation.
  /// If complete, destroys and nulls the event and executes callbacks, similarly to having called `wait`.
  /// @return True if the request has completed or is null/inactive, false otherwise.
  [[nodiscard]] auto test() -> bool {
    if (request_ == nullptr) {
      execute_all_callbacks();
      return true;
    }
    cudaError_t err = cudaEventQuery(request_);
    if (err == cudaSuccess) {
      retire_event();
      execute_all_callbacks();
      return true;
    } else if (err == cudaErrorNotReady) {
      return false;
    }

    // FIXME: Do something smarter with `err` for better error reporting
    nccl::fail_if(err != cudaSuccess, "KokkosComm::Request::test: request query failed");
    // unreachable
    return false;
  }

 private:
  request_type request_;
  std::vector<std::function<void()>> callbacks_;

  auto retire_event() -> void {
    nccl::fail_if(cudaEventDestroy(request_) != cudaSuccess, "KokkosComm::Request: completed event destruction failed");
    request_ = nullptr;
  }

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
/// Retires all active events before executing any completion callbacks.
/// @param requests The list of requests to complete.
inline auto wait_all(std::span<Request<Experimental::NcclSpace>> requests) -> void {
  size_t remaining = 0;
  for (const auto& req : requests) {
    remaining += req.request() != nullptr;
  }

  // Poll without extra tracking storage. Retiring an event nulls its handle, so each completion is counted once.
  while (remaining > 0) {
    for (auto& req : requests) {
      if (req.request() == nullptr) continue;
      cudaError_t err = cudaEventQuery(req.request());
      if (err == cudaSuccess) {
        req.retire_event();
        remaining--;
      } else {
        nccl::fail_if(err != cudaErrorNotReady, "KokkosComm::Request::wait_all: request completions failed");
      }
    }
  }
  for (auto& req : requests) {
    req.execute_all_callbacks();
  }
}

/// @brief Waits for the completion of one request among all passed requests.
/// @param requests The list of requests to try to complete.
/// @return The selected index, or `std::nullopt` if the span is empty or all handles are inactive.
inline auto wait_any(std::span<Request<Experimental::NcclSpace>> requests)
    -> std::optional<typename Request<Experimental::NcclSpace>::rank_type> {
  // Poll active events only; a retired/null event must never be selected again.
  while (true) {
    bool has_active = false;
    for (size_t r = 0; r < requests.size(); ++r) {
      if (requests[r].request() == nullptr) continue;
      has_active      = true;
      cudaError_t err = cudaEventQuery(requests[r].request());
      if (err == cudaSuccess) {
        requests[r].retire_event();
        requests[r].execute_all_callbacks();
        return static_cast<typename Request<Experimental::NcclSpace>::rank_type>(r);
      } else {
        nccl::fail_if(err != cudaErrorNotReady, "KokkosComm::Request::wait_any: request completion failed");
      }
    }
    if (!has_active) return std::nullopt;
  }
}

/// @brief Queries the request for completion of the associated operation.
/// @param request A reference on the request to query its completion.
inline auto test(Request<Experimental::NcclSpace>& request) -> bool { return request.test(); }

}  // namespace KokkosComm
