// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <memory>
#include <mutex>

#include <KokkosComm/config.hpp>

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
using DeviceStream = cudaStream_t;
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
using DeviceStream = hipStream_t;
#endif

namespace test_utils {

class XcclCtx {
 public:
  ~XcclCtx();
  // Singleton context; access through get().
  XcclCtx(const XcclCtx&)                    = delete;
  auto operator=(const XcclCtx&) -> XcclCtx& = delete;
  XcclCtx(XcclCtx&&)                         = delete;
  auto operator=(XcclCtx&&) -> XcclCtx&      = delete;

  static auto init(bool verbose = true) -> void;
  static auto fini() -> void;
  static auto get() -> XcclCtx&;

  auto comm() const -> ncclComm_t;
  auto stream() const -> DeviceStream;
  auto size() const -> int;
  auto rank() const -> int;
  auto device() const -> int;

 private:
  XcclCtx(ncclComm_t comm, DeviceStream stream, int dev, int n_ranks, int my_rank);

  ncclComm_t comm_     = nullptr;
  DeviceStream stream_ = nullptr;
  int dev_             = -1;
  int size_            = 0;
  int rank_            = 0;

  static std::unique_ptr<XcclCtx> instance_;
  static std::once_flag init_flag_;
};

}  // namespace test_utils
