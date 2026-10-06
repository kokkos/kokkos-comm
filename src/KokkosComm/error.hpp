#pragma once

#include <tl/expected.hpp>

namespace KokkosComm {

enum ErrorCode {
  NoError,
  NotSupported,
  BackendError
};  // Backend error: error comes from the backend i.e. MPI or
    // nccl.

struct Error {
  ErrorCode code;
  std::optional<int> backend_code{};  // only when error came from MPI/NCCL
};

using status_type = tl::expected<void, Error>;

}  // namespace KokkosComm
