#pragma once

#include <tl/expected.hpp>
#include <optional>

namespace KokkosComm {

enum class ErrorCode { NoError, NotSupported, BackendError };  // Backend error: MPI or nccl encountered and error.

struct Error {
  ErrorCode code;                   // only when an error has occurred
  std::optional<int> backend_code;  // only when error came from MPI/NCCL
};

using status_type = tl::expected<void, Error>;

}  // namespace KokkosComm
