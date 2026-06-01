// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <memory>
#include <mutex>

#include <nccl.h>
#include <cuda_runtime.h>

namespace test_utils {

class NcclCtx {
 public:
  ~NcclCtx();
  // Singleton context; access through get().
  NcclCtx(const NcclCtx&)                    = delete;
  auto operator=(const NcclCtx&) -> NcclCtx& = delete;
  NcclCtx(NcclCtx&&)                         = delete;
  auto operator=(NcclCtx&&) -> NcclCtx&      = delete;

  static auto init(bool verbose = true) -> void;
  static auto fini() -> void;
  static auto get() -> NcclCtx&;

  auto comm() const -> ncclComm_t;
  auto stream() const -> cudaStream_t;
  auto size() const -> int;
  auto rank() const -> int;
  auto device() const -> int;

 private:
  NcclCtx(ncclComm_t comm, cudaStream_t stream, int dev, int n_ranks, int my_rank);

  ncclComm_t comm_     = nullptr;
  cudaStream_t stream_ = nullptr;
  int dev_             = -1;
  int size_            = 0;
  int rank_            = 0;

  static std::unique_ptr<NcclCtx> instance_;
  static std::once_flag init_flag_;
};

}  // namespace test_utils
