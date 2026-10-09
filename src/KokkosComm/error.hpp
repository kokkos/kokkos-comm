#pragma once

#include <tl/expected.hpp>
#include <iostream>
#include <optional>
#include <string_view>

namespace KokkosComm {

/// @brief Category of an error reported by Kokkos Comm.
enum class ErrorCode {
  NoError,       ///< No error occurred.
  NotSupported,  ///< The requested operation is not supported for the given arguments (e.g., non-contiguous views).
  MpiError,
  NcclError,
  CudaError
};

/// @brief Describes an error reported by Kokkos Comm.
struct Error {
  /// @brief Category of the error.
  ErrorCode code;
  /// @brief Raw error code returned by the backend, only when relevant
  std::optional<int> backend_code;
};

/// @brief Result of a Kokkos Comm operation: empty on success, holds an `Error` on failure.
using status_type = tl::expected<void, Error>;

namespace Impl {

/// @brief Prints a diagnostic for a failed KokkosComm-level check (e.g. an unsupported argument).
/// @param cond The text of the condition that triggered the failure.
inline auto print_check_failed(std::string_view cond, const char* file, int line) -> void {
  std::cerr << "Error: KokkosComm check `" << cond << "` failed at " << file << ":" << line << std::endl;
}

}  // namespace Impl

}  // namespace KokkosComm
