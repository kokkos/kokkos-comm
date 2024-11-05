//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#pragma once

#include <KokkosComm/fwd.hpp>
#include <KokkosComm/mpi/mpi.hpp>

#include <vector>
#include <utility>
#include <functional>

namespace KokkosComm {

template <>
class Req<Mpi> {
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

  friend void wait(Req<Mpi> req);
  friend void wait_all(std::vector<Req<Mpi>> &reqs);
  friend int wait_any(std::vector<Req<Mpi>> &reqs);
};

inline void wait(Req<Mpi> req) {
  MPI_Wait(&req.mpi_request(), MPI_STATUS_IGNORE);
  for (auto &f : req.record_->postWaits_) {
    f();
  }
  req.record_->postWaits_.clear();
}

inline void wait_all(std::vector<Req<Mpi>> &reqs) {
  for (Req<Mpi> &req : reqs) {
    wait(req);
  }
}

inline int wait_any(std::vector<Req<Mpi>> &reqs) {
  for (size_t i = 0; i < reqs.size(); ++i) {
    int completed;
    MPI_Test(&(reqs[i].mpi_request()), &completed, MPI_STATUS_IGNORE);
    if (completed) {
      return true;
    }
  }
  return false;
}

}  // namespace KokkosComm
