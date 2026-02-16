// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <functional>
#include <memory>
#include <span>
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

  /// @brief Constructs a `Request` from an `MPI_Request`.
  /// @param request The request to encapsulate. Defaults to `MPI_REQUEST_NULL`.
  explicit Request(request_type request = MPI_REQUEST_NULL) : request_(request) {}
  /// @brief Destructor.
  ~Request() = default;

  /// @brief Copy constructor is deleted because a `Request` can only be moved.
  Request(const Request&) = delete;
  /// @brief Copy assignment operator is deleted because a `Request` can only be moved.
  auto operator=(const Request&) -> Request& = delete;
  /// @brief Move constructor.
  Request(Request&&) = default;
  /// @brief Move assignment operator.
  auto operator=(Request&&) -> Request& = default;

  /// @return A reference to the underlying `MPI_Request` object.
  [[nodiscard]] constexpr auto request() noexcept -> request_type& { return request_; }
  /// @return A const reference to the underlying `MPI_Request` object.
  [[nodiscard]] constexpr auto request() const noexcept -> const request_type& { return request_; }
  /// @return A pointer to the underlying `MPI_Request` object.
  [[nodiscard]] constexpr auto request_ptr() noexcept -> request_type* { return &request_; }
  /// @return A const pointer to the underlying `MPI_Request` object.
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
  /// The underlying `MPI_Request` object is set to `MPI_REQUEST_NULL` upon return.
  auto wait() -> void {
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
/// Incurs an overhead for copying the underlying `MPI_Request` objects to an intermediate container.
/// @param requests The list of requests to complete.
inline auto wait_all(std::span<Request<MpiSpace>> requests) -> void {
  if (requests.empty()) {
    return;
  }

  std::vector<MPI_Request> mpi_requests;
  std::vector<MPI_Status> mpi_statuses;
  mpi_requests.reserve(requests.size());
  mpi_statuses.reserve(requests.size());
  for (auto& req : requests) {
    mpi_requests.push_back(req.request());
  }
  int err = MPI_Waitall(static_cast<int>(mpi_requests.size()), mpi_requests.data(), mpi_statuses.data());
  // FIXME: Do something smarter with `statuses` for better error handling and reporting
  mpi::fail_if(err != MPI_SUCCESS, "KokkosComm::Request::wait_all: request completions failed");

  for (auto& req : requests) {
    req.execute_all_callbacks();
  }
}

/// @brief Waits for the completion of one request among all passed requests.
/// Incurs an overhead for copying the underlying `MPI_Request` objects to an intermediate container.
/// @param requests The list of requests to try to complete.
/// @return The index of the request within the passed list upon successful completion, `std::nullopt` otherwise.
inline auto wait_any(std::span<Request<MpiSpace>> requests) -> std::optional<typename Request<MpiSpace>::rank_type> {
  if (requests.empty()) {
    return std::nullopt;
  }

  std::vector<MPI_Request> mpi_requests(requests.size());
  for (auto& req : requests) {
    mpi_requests.push_back(req.request());
  }
  int idx;
  MPI_Status status;
  int err = MPI_Waitany(static_cast<int>(mpi_requests.size()), mpi_requests.data(), &idx, &status);
  // FIXME: Do something smarter with `status` for better error handling and reporting
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
