#pragma once

#include <mpi.h>

namespace KokkosComm {

class Req {
public:
  Req() : req_(MPI_REQUEST_NULL) {}

  MPI_Request &mpi_req() { return req_; }

  void wait() { MPI_Wait(&req_, MPI_STATUS_IGNORE); }

private:
  MPI_Request req_;
};

} // namespace KokkosComm