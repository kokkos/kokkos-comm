// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>
#include <optional>

#include <cuda_runtime.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/fwd.hpp>
#include <KokkosComm/error.hpp>

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
  using status_type         = KokkosComm::status_type;

  /// @brief Constructs a `Request`.
  explicit Request() : request_(nullptr), comm_({}) {}

  /// @brief Constructs a `Request`.
  /// @param comm The communicator used in the request.
  explicit Request(ncclComm_t comm) : request_(nullptr), comm_(comm) {}

  /// @brief Constructs a `Request` in the failed state, without any associated operation.
  /// Used when posting the operation failed; `wait` on such a request returns immediately.
  /// @param e The error to store in the request.
  /// @return A `Request` for which `has_error()` is true.
  static auto failed(Error e) -> Request {
    Request r;  // request_ = nullptr
    r.status_ = tl::unexpected(e);
    return r;
  }

  /// @brief Capture the state of a `cudaStream_t` for request encapsulation.
  /// @param stream The stream to capture for request encapsulation.
  [[nodiscard]] auto capture_stream_state(cudaStream_t stream) noexcept -> cudaError_t {
    cudaError_t err;
    if (request_ != nullptr) {
      err = cudaEventDestroy(request_);
      if (err != cudaSuccess) return err;
    }
    err = cudaEventCreate(&request_, cudaEventDisableTiming);
    if (err != cudaSuccess) return err;
    err = cudaEventRecord(request_, stream);
    return err;
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
  /// @return The category of the error stored in the request, or `ErrorCode::NoError` if there is none.
  /// Can be queried before `wait` to detect a failure to post the operation.
  [[nodiscard]] auto error_code() const noexcept -> ErrorCode {
    return !has_error() ? KokkosComm::ErrorCode::NoError : status_.error().code;
  }
  /// @return The raw backend error code stored in the request, if the error came from the backend.
  [[nodiscard]] auto backend_error_code() const noexcept -> std::optional<int> {
    return !has_error() ? std::nullopt : status_.error().backend_code;
  }

  /// @return True if posting or completing the associated operation failed, false otherwise.
  [[nodiscard]] auto has_error() const noexcept -> bool { return !status_.has_value(); }

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
  /// If the request is already in the failed state, returns immediately. If `cudaEventSynchronize` fails, the error
  /// is stored in the request with the `cudaError_t` as backend code. In both cases, registered callbacks are discarded
  /// without being invoked; check `has_error()` after waiting.
  auto wait() -> void {
    if (has_error()) {     // post already failed: no event was recorded, don't touch CUDA
      callbacks_.clear();  // drop lifetime captures, skip unpack
      return;
    }

    while (true) {
      // poll the status of the request (event)
      cudaError_t cuda_err = cudaEventQuery(request_);
      if (cuda_err == cudaSuccess) break;

      // If it's not success, it might just not be ready yet (cudaErrorNotReady).
      // If it failed, return
      if (cuda_err != cudaErrorNotReady) {
        status_ = tl::unexpected(Error{KokkosComm::ErrorCode::CudaError, static_cast<int>(cuda_err)});
        callbacks_.clear();
        return;
      }

      // If it's not ready yet, it could be because of a nccl async error, we poll it, but first check that poll
      ncclResult_t async_err = ncclSuccess;
      if (ncclResult_t nccl_err = ncclCommGetAsyncError(comm_, &async_err); nccl_err != ncclSuccess) {
        status_ = tl::unexpected(Error{KokkosComm::ErrorCode::NcclError, static_cast<int>(nccl_err)});
        if (nccl_err == ncclUnhandledCudaError) {
          nccl::print_cuda_error_hint();
        }
        callbacks_.clear();
        return;
      }

      // If we did find an error, we return
      // N.B. it can't be ncclInProgress
      if (async_err != ncclSuccess) {
        status_ = tl::unexpected(Error{KokkosComm::ErrorCode::NcclError, static_cast<int>(async_err)});
        if (async_err == ncclUnhandledCudaError) {
          nccl::print_cuda_error_hint();
        }
        callbacks_.clear();
        return;
      }
    }

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
  status_type status_{};  // no error by default
  ncclComm_t comm_;       // Copy of the nccl communicator used for the request, to poll errors only

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
