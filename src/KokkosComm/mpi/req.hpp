// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <functional>
#include <span>
#include <vector>

#include <mpi.h>

#include <KokkosComm/fwd.hpp>
#include "mpi_space.hpp"

namespace KokkosComm {

template <>
class Req<MpiSpace> {
  // a type-erased view. Request uses these to keep temporary views alive for
  // the lifetime of "Immediate" MPI operations
  struct ViewHolderBase {
    virtual ~ViewHolderBase() {}
  };
  template <typename V>
  struct ViewHolder : ViewHolderBase {
    ViewHolder(const V &v) : v_(v) {}
    V v_;
  };

  struct Record {
    Record() : req_(MPI_REQUEST_NULL) {}
    MPI_Request req_;
    std::vector<std::function<void()>> postWaits_;
  };

 public:
  Req() : record_(std::make_shared<Record>()) {}

  MPI_Request &mpi_request() { return record_->req_; }

  // keep a reference to this view around until wait() is called
  template <typename View>
  void extend_view_lifetime(const View &v) {
    // unmanaged views don't own the underlying buffer, so no need to extend lifetime
    if (v.use_count() != 0) {
      record_->postWaits_.push_back([v]() {});
    }
  }

  void call_after_mpi_wait(std::function<void()> &&f) { record_->postWaits_.push_back(f); }

 private:
  std::shared_ptr<Record> record_;

  friend void wait(Req<MpiSpace> &req);
  friend void wait(Req<MpiSpace> &&req);
  friend void wait_all(std::span<Req<MpiSpace>> reqs);
  friend void wait_any(std::span<Req<MpiSpace>> reqs);
};

inline void wait(Req<MpiSpace> &req) {
  MPI_Wait(&req.mpi_request(), MPI_STATUS_IGNORE);
  for (auto &f : req.record_->postWaits_) {
    f();
  }
  req.record_->postWaits_.clear();
}

inline void wait(Req<MpiSpace> &&req) { wait(req); }

inline void wait_all(std::span<Req<MpiSpace>> reqs) {
  for (Req<MpiSpace> &req : reqs) {
    wait(req);
  }
}

/// FIXME: This function will loop indefinitely if all requests in the list are equivalent to `MPI_REQUEST_NULL`.
/// FIXME: This function should return the index of the completed request, if any.
inline void wait_any(std::span<Req<MpiSpace>> reqs) {
  // FIXME: Active wait-loop
  while (true) {
    for (Req<MpiSpace> &req : reqs) {
      int flag;
      MPI_Test(&(req.mpi_request()), &flag, MPI_STATUS_IGNORE);
      if (flag) {
        wait(req);
        return;
      }
    }
  }
}

}  // namespace KokkosComm
