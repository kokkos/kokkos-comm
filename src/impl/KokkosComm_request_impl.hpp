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

#include <memory>

#include "KokkosComm_include_mpi.hpp"

namespace KokkosComm::Impl {

class Req {
  // a type-erased callable. Req uses these to attach callbacks to be executed
  // at wait
  struct InvokableHolderBase {
    virtual ~InvokableHolderBase() = default;

    virtual void operator()() = 0;
  };
  template <Invokable Fn>
  struct InvokableHolder : InvokableHolderBase {
    InvokableHolder(const Fn &f) : f_(f) {}

    virtual void operator()() override { f_(); }

    Fn f_;
  };

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

 public:
  Req() : req_(MPI_REQUEST_NULL) {}

  MPI_Request &mpi_req() { return req_; }

  void wait() {
    MPI_Wait(&req_, MPI_STATUS_IGNORE);
    wait_drops_.clear();  // drop any views we're keeping alive until wait()
    for (auto &c : wait_callbacks_) {
      (*c)();
    }
    wait_callbacks_.clear();
  }

  // keep a reference to this view around until wait() is called
  template <typename View>
  void keep_until_wait(const View &v) {
    wait_drops_.push_back(std::make_shared<ViewHolder<View>>(v));
  }

  template <Invokable Fn>
  void call_and_drop_at_wait(const Fn &f) {
    wait_callbacks_.push_back(std::make_shared<InvokableHolder<Fn>>(f));
  }

 private:
  MPI_Request req_;
  std::vector<std::shared_ptr<ViewHolderBase>> wait_drops_;
  std::vector<std::shared_ptr<InvokableHolderBase>> wait_callbacks_;
};

}  // namespace KokkosComm::Impl
