#pragma once

#include <tl/expected.hpp>

namespace KokkosComm {

enum ErrorCode { WrongExtent, KokkosNotInitialized, MPIError, NCCLError };

struct Error {
  ErrorCode code;
  int backend_code;
};

using status_type = tl::expected<void, Error>;

}  // namespace KokkosComm
