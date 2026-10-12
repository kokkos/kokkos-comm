// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <mpi.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/fwd.hpp>
#include "mpi_space.hpp"

#include "impl/error_handling.hpp"

namespace KokkosComm {

/// @brief Request specialization for the MPI communication space.
template <>
class Request<MpiSpace> {
 public:
  using communication_space = MpiSpace;
  using request_type        = MpiSpace::request_type;
  using rank_type           = MpiSpace::rank_type;

  /// @brief Constructs an inactive request.
  Request() noexcept : request_(MPI_REQUEST_NULL) {}

  /// @brief Adopts a non-persistent MPI request, leaving the source null.
  /// Only the handle is transferred; the caller must keep all associated buffers and resources valid until completion.
  /// Persistent MPI requests are not supported; use Channel<MpiSpace> for persistent communication.
  explicit Request(request_type&& request) noexcept : request_(std::exchange(request, MPI_REQUEST_NULL)) {}

  /// @brief Requires explicit completion before destruction; never waits, cancels, or frees an operation.
  ~Request() noexcept {
    mpi::fail_if(
        request_ != MPI_REQUEST_NULL || !callbacks_.empty(), "KokkosComm::Request: destruction requires completion"
    );
  }

  /// @brief Copy constructor is deleted because a `Request` can only be moved.
  Request(const Request&) = delete;
  /// @brief Copy assignment operator is deleted because a `Request` can only be moved.
  auto operator=(const Request&) -> Request& = delete;
  /// @brief Move constructor.
  Request(Request&& other) noexcept
      : request_(std::exchange(other.request_, MPI_REQUEST_NULL)), callbacks_(std::exchange(other.callbacks_, {})) {}
  /// @brief Move assignment operator.
  /// Requires an inactive destination with no pending callbacks. Self-move is a no-op.
  auto operator=(Request&& other) noexcept -> Request& {
    if (this != &other) {
      mpi::fail_if(
          request_ != MPI_REQUEST_NULL || !callbacks_.empty(),
          "KokkosComm::Request: move assignment requires an inactive destination"
      );
      request_   = std::exchange(other.request_, MPI_REQUEST_NULL);
      callbacks_ = std::exchange(other.callbacks_, {});
    }
    return *this;
  }

  /// @brief Borrowed native access. Applications must not complete, free, replace, or retain an owning handle copy.
  /// Backend implementations may initialize an inactive request through this access.
  /// @return A reference to the underlying `MPI_Request` object.
  [[nodiscard]] constexpr auto request() noexcept -> request_type& { return request_; }
  /// @return A const reference to the underlying `MPI_Request` object.
  [[nodiscard]] constexpr auto request() const noexcept -> const request_type& { return request_; }
  /// @return A pointer to the underlying `MPI_Request` object.
  [[nodiscard]] constexpr auto request_ptr() noexcept -> request_type* { return &request_; }
  /// @return A const pointer to the underlying `MPI_Request` object.
  [[nodiscard]] constexpr auto request_ptr() const noexcept -> const request_type* { return &request_; }

  /// @brief Implementation hook: registers completion work; applications must not register callbacks.
  /// Callbacks must be nonempty, must not throw, and must not mutate, reactivate, or re-enter their owning request.
  /// @param cb The callback function to register.
  auto add_callback(std::function<void()>&& cb) -> void { callbacks_.push_back(std::move(cb)); }

  /// @brief Captures a Kokkos View to extend its lifetime until the request's completion.
  /// Managed allocations are retained until completion. For unmanaged Views, the application must keep the
  /// underlying user-owned allocation valid until completion; Kokkos Comm cannot retain it.
  /// @tparam V A Kokkos View type.
  /// @param view The Kokkos View to capture for lifetime extension.
  template <KokkosView V>
  auto extend_view_lifetime(const V& view) -> void {
    // Unmanaged Views do not own their allocation, so Kokkos Comm cannot retain it.
    if (view.use_count() != 0) {
      add_callback([view]() {});
    }
  }

  /// @brief Waits on the request until completion of the associated operation.
  /// The underlying `MPI_Request` object is set to `MPI_REQUEST_NULL` upon return.
  auto wait() -> void {
    if (request_ == MPI_REQUEST_NULL) {
      execute_all_callbacks();
      return;
    }
    MPI_Status status;
    int err = MPI_Wait(request_ptr(), &status);
    // FIXME: Do something smarter with status` for better error handling and reporting
    mpi::fail_if(err != MPI_SUCCESS, "KokkosComm::Request::wait: request completion failed");

    execute_all_callbacks();
  }

  /// @brief Queries the request for the completion of the associated operation.
  /// If the operation has completed, all callbacks are executed and the underlying `MPI_Request` object is set to
  /// `MPI_REQUEST_NULL` upon return, similarly to having called `wait`.
  /// @return True if the request has completed or is null/inactive, false otherwise.
  [[nodiscard]] auto test() -> bool {
    if (request_ == MPI_REQUEST_NULL) {
      execute_all_callbacks();
      return true;
    }
    int has_completed;
    MPI_Status status;
    int err = MPI_Test(request_ptr(), &has_completed, &status);
    // FIXME: Do something smarter with status` for better error handling and reporting
    mpi::fail_if(err != MPI_SUCCESS, "KokkosComm::Request::test: request query failed");

    if (has_completed) {
      execute_all_callbacks();
    }
    return static_cast<bool>(has_completed);
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
inline auto wait(Request<MpiSpace>& request) -> void { request.wait(); }
/// @brief Waits on the request until completion of the associated operation.
/// @param request An r-value reference on the request, consumed upon completion.
inline auto wait(Request<MpiSpace>&& request) -> void { request.wait(); }

/// @brief Waits for the completion of all passed requests.
/// Temporarily transfers the native handles to contiguous storage; callbacks run after all operations complete.
/// @param requests The list of requests to complete.
inline auto wait_all(std::span<Request<MpiSpace>> requests) -> void {
  if (requests.empty()) {
    return;
  }

  std::vector<MPI_Request> mpi_requests;
  mpi_requests.reserve(requests.size());
  for (auto& req : requests) {
    mpi_requests.push_back(std::exchange(req.request_, MPI_REQUEST_NULL));
  }
  int err = MPI_Waitall(static_cast<int>(mpi_requests.size()), mpi_requests.data(), MPI_STATUSES_IGNORE);
  mpi::fail_if(err != MPI_SUCCESS, "KokkosComm::Request::wait_all: request completions failed");

  for (auto& req : requests) {
    req.execute_all_callbacks();
  }
}

/// @brief Waits for the completion of one request among all passed requests.
/// Temporarily transfers the native handles to contiguous storage and restores all non-selected handles.
/// @param requests The list of requests to try to complete.
/// @return The selected index, or `std::nullopt` if the span is empty or all handles are inactive.
inline auto wait_any(std::span<Request<MpiSpace>> requests) -> std::optional<typename Request<MpiSpace>::rank_type> {
  if (requests.empty()) {
    return std::nullopt;
  }

  std::vector<MPI_Request> mpi_requests;
  mpi_requests.reserve(requests.size());
  for (auto& req : requests) {
    mpi_requests.push_back(std::exchange(req.request_, MPI_REQUEST_NULL));
  }
  int idx;
  int err = MPI_Waitany(static_cast<int>(mpi_requests.size()), mpi_requests.data(), &idx, MPI_STATUS_IGNORE);
  // MPI nulls only the selected handle. Restore ownership before running its callbacks.
  for (size_t r = 0; r < requests.size(); ++r) {
    requests[r].request_ = std::exchange(mpi_requests[r], MPI_REQUEST_NULL);
  }
  mpi::fail_if(err != MPI_SUCCESS, "KokkosComm::Request::wait_any: request completion failed");

  if (idx == MPI_UNDEFINED) {
    return std::nullopt;
  }

  requests[idx].execute_all_callbacks();
  return static_cast<typename Request<MpiSpace>::rank_type>(idx);
}

/// @brief Queries the request for completion of the associated operation.
/// @param request A reference on the request to query its completion.
inline auto test(Request<MpiSpace>& request) -> bool { return request.test(); }

}  // namespace KokkosComm
