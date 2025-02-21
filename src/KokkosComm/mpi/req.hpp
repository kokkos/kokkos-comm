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

  template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
  friend struct KokkosComm::Impl::Wait;

  template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
  friend struct KokkosComm::Impl::WaitAll;

  template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
  friend struct KokkosComm::Impl::WaitAny;
};

}  // namespace KokkosComm