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

  // a type-erased view. Request uses these to keep temporary views alive for
  // the lifetime of "Immediate" MPI operations
  struct SpaceHolderBase {
    virtual ~SpaceHolderBase() {}

    virtual void fence() = 0;
  };
  template <typename ES>
  struct SpaceHolder : SpaceHolderBase {
    SpaceHolder(const ES &es) : es_(es) {}

    virtual void fence() override { es_.fence("KokkosComm::Req::wait()"); }

    ES es_;
  };

 public:
  Req() : req_(MPI_REQUEST_NULL) {}

  MPI_Request &mpi_req() { return req_; }

  // The communication must be done before the held execution space can do any
  // further work. For MPI, this is achieved by blocking the host thread and
  // fencing the execution space.
  void wait() {
    MPI_Wait(&req_, MPI_STATUS_IGNORE);
    for (auto &c : wait_callbacks_) {
      (*c)();
    }

    // drop the references to anything that was kept alive until wait
    wait_drops_.clear();
    wait_callbacks_.clear();

    if (exec_space_) {
      exec_space_->fence();
    }
    exec_space_ = nullptr;
  }

  // Keep a reference to this view around until wait() is called.
  // This is used when a managed Kokkos::View is provided to an
  // asychronous communication routine, to ensure that view is
  // still alive for the entire duration of the routine.
  template <typename View>
  void keep_until_wait(const View &v) {
    wait_drops_.push_back(std::make_shared<ViewHolder<View>>(v));
  }

  // When wait() is called: execute f() and then let f go out of scope.
  // Every stored f is called before any stored f is dropped.
  // This function can be used by an unpacking routine to attach some
  // unpacking logic to a communication that needs to be executed
  // after the underlying MPI operation is done.
  template <Invokable Fn>
  void call_and_drop_at_wait(const Fn &f) {
    wait_callbacks_.push_back(std::make_shared<InvokableHolder<Fn>>(f));
  }

  template <typename ExecSpace>
  void fence_at_wait(const ExecSpace &space) {
    if (exec_space_) {
      Kokkos::abort("Req is already fencing a space!");
    }
    exec_space_ = std::make_shared<SpaceHolder<ExecSpace>>(space);
  }

 private:
  MPI_Request req_;
  std::vector<std::shared_ptr<ViewHolderBase>> wait_drops_;
  std::vector<std::shared_ptr<InvokableHolderBase>> wait_callbacks_;
  std::shared_ptr<SpaceHolderBase> exec_space_;
};

}  // namespace KokkosComm::Impl
