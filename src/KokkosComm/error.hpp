#pragma once

#include <tl/expected.hpp>

namespace KokkosComm {

enum ErrorCode { NotSupported, MPIError, NCCLError };

struct Error {
  ErrorCode code;
  int backend_code;
};

using status_type = tl::expected<void, Error>;

}  // namespace KokkosComm
