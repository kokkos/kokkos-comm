// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <cuda.h>

#include <KokkosComm/fwd.hpp>
#include "nccl_space.hpp"

#include "impl/cuda_check.hpp"

namespace KokkosComm {

template <>
class Req<Experimental::NcclSpace> {
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
    explicit Record(cudaStream_t stream) : req_(stream) {}

    cudaStream_t req_;
    std::vector<std::function<void()>> postWaits_;
  };

 public:
  explicit Req(cudaStream_t stream) : record_(std::make_shared<Record>(stream)) {}

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

  friend void wait(Req<Experimental::NcclSpace> &req);
  friend void wait(Req<Experimental::NcclSpace> &&req);
  friend void wait_all(std::span<Req<Experimental::NcclSpace>> reqs);
  friend int wait_any(std::span<Req<Experimental::NcclSpace>> reqs);
};

inline auto wait(Req<Experimental::NcclSpace> &req) -> void {
  KC_CUDA_CHECK(cudaStreamSynchronize(req.get_inner()));
  for (auto &f : req.record_->postWaits_) {
    f();
  }
  req.record_->postWaits_.clear();
}

inline auto wait(Req<Experimental::NcclSpace> &&req) -> void { wait(req); }

inline auto wait_all(std::span<Req<Experimental::NcclSpace>> reqs) -> void {
  for (Req<Experimental::NcclSpace> &req : reqs) {
    wait(req);
  }
}

inline auto wait_any(std::span<Req<Experimental::NcclSpace>> reqs) -> int {
  // Loop while we don't have at least one completed request.
  while (true) {
    for (size_t r = 0; r < reqs.size(); ++r) {
      cudaError_t res = cudaStreamQuery(reqs[r].get_inner());
      // If the current request has completed, we must make sure the post-wait callbacks run and are cleared.
      // Calling `wait` should be a no-op if the request has no callback to execute.
      if (res == cudaSuccess) {
        wait(reqs[r]);
        return static_cast<int>(r);
      } else if (res == cudaErrorNotReady) {
        continue;
      } else {
        throw std::runtime_error(cudaGetErrorString(res));
      }
    }
  }
}

}  // namespace KokkosComm
