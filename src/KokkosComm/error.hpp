#pragma once

namespace KokkosComm {

enum ErrorCode { WrongExtent, KokkosNotInitialized, MPIError };

struct Error {
  ErrorCode code;
  int backend_code;
};

}  // namespace KokkosComm
