#pragma once

#include <memory>

#include <mpi.h>

namespace KokkosComm {

class Req {
private:
  // a type-erased view. Request uses these to keep temporary views alive for
  // the lifetime of "Immediate" MPI operations
  struct ViewHolderBase {
    virtual ~ViewHolderBase() {}
  };
  template <typename V> struct ViewHolder : ViewHolderBase {
    ViewHolder(const V &v) : v_(v) {}
    V v_;
  };

public:
  Req() : req_(MPI_REQUEST_NULL) {}

  MPI_Request &mpi_req() { return req_; }

  void wait() {
    MPI_Wait(&req_, MPI_STATUS_IGNORE);
    until_waits_.clear(); // drop any views we're keeping alive until wait()
  }

  // keep a reference to this view around until wait() is called
  template <typename View> void keep_until_wait(const View &v) {
    until_waits_.push_back(std::make_shared<ViewHolder<View>>(v));
  }

private:
  MPI_Request req_;

  std::vector<std::shared_ptr<ViewHolderBase>> until_waits_;
};

} // namespace KokkosComm