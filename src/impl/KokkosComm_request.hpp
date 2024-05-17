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

#include "impl/KokkosComm_ViewHolder.hpp"
#include "impl/KokkosComm_InvokableHolder.hpp"

namespace KokkosComm::Impl {

// FIXME: this thing won't really work well on device
class Req {
  // a type-erased view. Request uses these to keep temporary views alive for
  // the lifetime of "Immediate" MPI operations

  struct Record {
    Record() = default;
    std::vector<MPI_Request> reqs_;
    std::vector<std::shared_ptr<ViewHolderBase>> until_waits_;
    std::vector<std::shared_ptr<InvokableHolderBase>> wait_callbacks_;
  };

 public:
  Req() : record_(std::make_shared<Record>()) {}

  void wait() {
    MPI_Waitall(record_->reqs_.size(), record_->reqs_.data(), MPI_STATUSES_IGNORE);
    record_->until_waits_.clear();  // drop any views we're keeping alive until wait()
    for (auto &c : record_->wait_callbacks_) {
      (*c)();
    }
    record_->wait_callbacks_.clear();
  }

  // keep a reference to this view around until wait() is called
  void keep_until_wait(const std::shared_ptr<ViewHolderBase> &vhb) { record_->until_waits_.push_back(vhb); }
  template <typename View>
  void keep_until_wait(const View &v) {
    keep_until_wait(std::make_shared<ViewHolderBase>(ViewHolder(v)));
  }

  void call_and_drop_at_wait(const std::shared_ptr<InvokableHolderBase> &ihb) {
    record_->wait_callbacks_.push_back(ihb);
  }

  MPI_Request add_mpi_wait() {
    record_->reqs_.push_back(MPI_Request{});
    return record_->reqs_.back();
  }
  MPI_Request add_mpi_wait(MPI_Request req) {
    record_->reqs_.push_back(req);
    return req;
  }

 private:
  std::shared_ptr<Record> record_;
};

}  // namespace KokkosComm::Impl