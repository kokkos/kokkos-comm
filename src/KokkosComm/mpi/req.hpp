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

#include <vector>
#include <utility>
#include <functional>

#include "KokkosComm/fwd.hpp"

#include "mpi.hpp"

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

  template <KokkosExecutionSpace ExecSpace>
  friend void wait(const ExecSpace &space, Req<Mpi> req);
  friend void wait(Req<Mpi> req);
  template <KokkosExecutionSpace ExecSpace>
  friend void wait_all(const ExecSpace &space, std::vector<Req<Mpi>> &reqs);
  friend void wait_all(std::vector<Req<Mpi>> &reqs);
  template <KokkosExecutionSpace ExecSpace>
  friend int wait_any(const ExecSpace &space, std::vector<Req<Mpi>> &reqs);
  friend int wait_any(std::vector<Req<Mpi>> &reqs);
};

template <KokkosExecutionSpace ExecSpace>
void wait(const ExecSpace &space, Req<Mpi> req) {
  /* Semantically this only guarantees that `space` is waiting for request to complete. For the MPI host API, we have no
   * choice but to fence the space before waiting on the host.*/
  space.fence();
  MPI_Wait(&req.mpi_request(), MPI_STATUS_IGNORE);
  for (auto &f : req.record_->postWaits_) {
    f();
  }
  req.record_->postWaits_.clear();
}

inline void wait(Req<Mpi> req) { wait(Kokkos::DefaultExecutionSpace(), req); }

template <KokkosExecutionSpace ExecSpace>
void wait_all(const ExecSpace &space, std::vector<Req<Mpi>> &reqs) {
  space.fence();
  for (Req<Mpi> &req : reqs) {
    MPI_Wait(&req.mpi_request(), MPI_STATUS_IGNORE);
    for (auto &f : req.record_->postWaits_) {
      f();
    }
    req.record_->postWaits_.clear();
  }
}

inline void wait_all(std::vector<Req<Mpi>> &reqs) { wait_all(Kokkos::DefaultExecutionSpace(), reqs); }

template <KokkosExecutionSpace ExecSpace>
int wait_any(const ExecSpace &space, std::vector<Req<Mpi>> &reqs) {
  if (reqs.empty()) {
    return -1;
  }

  space.fence();
  while (true) {  // wait until something is done
    for (size_t i = 0; i < reqs.size(); ++i) {
      int completed;
      Req<Mpi> &req = reqs[i];
      MPI_Test(&(req.mpi_request()), &completed, MPI_STATUS_IGNORE);
      if (completed) {
        for (auto &f : req.record_->postWaits_) {
          f();
        }
        req.record_->postWaits_.clear();
        return i;
      }
    }
  }
}

inline int wait_any(std::vector<Req<Mpi>> &reqs) { return wait_any(Kokkos::DefaultExecutionSpace(), reqs); }

}  // namespace KokkosComm