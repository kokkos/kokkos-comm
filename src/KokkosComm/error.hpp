#pragma once

#include <tl/expected.hpp>

namespace KokkosComm {

enum ErrorCode { NotSupported, BackendError };  // Backend error: MPI or nccl encountered and error.

struct Error {
  std::optional<ErrorCode> code;    // only when an error has occurred
  std::optional<int> backend_code;  // only when error came from MPI/NCCL
};

using status_type = tl::expected<void, Error>;

}  // namespace KokkosComm
