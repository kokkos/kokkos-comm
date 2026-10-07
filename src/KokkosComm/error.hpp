#pragma once

#include <tl/expected.hpp>
#include <optional>

namespace KokkosComm {

/// @brief Category of an error reported by Kokkos Comm.
enum class ErrorCode {
  NoError,       ///< No error occurred.
  NotSupported,  ///< The requested operation is not supported for the given arguments (e.g., non-contiguous views).
  BackendError,  ///< The communication backend (MPI, NCCL or CUDA) reported an error.
};

/// @brief Describes an error reported by Kokkos Comm.
struct Error {
  /// @brief Category of the error.
  ErrorCode code;
  /// @brief Raw error code returned by the backend, only set when `code` is `ErrorCode::BackendError`.
  std::optional<int> backend_code;
};

/// @brief Result of a Kokkos Comm operation: empty on success, holds an `Error` on failure.
using status_type = tl::expected<void, Error>;

}  // namespace KokkosComm
