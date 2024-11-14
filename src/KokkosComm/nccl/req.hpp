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
#include <KokkosComm/nccl/nccl.hpp>

#include <cuda.h>

#include <span>
#include <utility>
#include <functional>

namespace KokkosComm::Experimental {

template <>
class Req<Nccl> {
  // A type-erased view. Request uses these to keep temporary views alive for the lifetime of NCCL operations
  struct ViewHolderBase {
    virtual ~ViewHolderBase() {}
  };

  template <typename V>
  struct ViewHolder : ViewHolderBase {
    ViewHolder(const V &v) : v_(v) {}
    V v_;
  };

  struct Record {
    Record() : req_() {}
    cudaStream_t req_;
    std::vector<std::function<void()>> postWaits_;
  };

 public:
  Req() : record_(std::make_shared<Record>()) {}

  auto get_inner() -> cudaStream_t & { return record_->req_; }

  // keep a reference to this view around until wait() is called
  template <typename View>
  auto extend_view_lifetime(const View &v) -> void {
    // unmanaged views don't own the underlying buffer, so no need to extend lifetime
    if (v.use_count() != 0) {
      record_->postWaits_.push_back([v]() {});
    }
  }

  auto call_after_wait(std::function<void()> &&f) -> void { record_->postWaits_.push_back(f); }

 private:
  std::shared_ptr<Record> record_;

  friend void wait(Req<Nccl> req);
  friend void wait_all(std::span<Req<Nccl>> reqs);
  friend int wait_any(std::span<Req<Nccl>> reqs);
};

inline auto wait(Req<Nccl> req) -> void {
  cudaStreamSynchronize(&req.get_inner());
  for (auto &f : req.record_->postWaits_) {
    f();
  }
  req.record_->postWaits_.clear();
}

inline auto wait_all(std::span<Req<Nccl>> reqs) -> void {
  for (Req<Nccl> &req : reqs) {
    wait(req);
  }
}

inline auto wait_any(std::span<Req<Nccl>> reqs) -> int {
  for (size_t i = 0; i < reqs.size(); ++i) {
    auto completed = cudaStreamQuery(&reqs[i].get_inner());
    if (cudaSuccess == completed) {
      return true;
    }
  }
  return false;
}

}  // namespace KokkosComm::Experimental
