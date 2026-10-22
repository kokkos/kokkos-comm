// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstdint>

#include <gtest/gtest.h>
#include <nccl.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/KokkosComm.hpp>

#include "utils.hpp"

#include <iostream>

namespace {

using ExecSpace = Kokkos::Cuda;
using CommSpace = KokkosComm::Experimental::NcclSpace;

TEST(ErrorHandling, ncclInvalidArgument) {
  auto& nccl_ctx  = test_utils::NcclCtx::get();
  const auto exec = Kokkos::Cuda(nccl_ctx.stream());
  const auto comm = nccl_ctx.comm();
  const int size  = nccl_ctx.size();
  const int rank  = nccl_ctx.rank();

  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }

  constexpr int src      = 0;
  constexpr int dst      = 1;
  constexpr int bad_rank = -1;

  Kokkos::View<int*> v("v", 42);

  if (rank == src) {
    KokkosComm::Experimental::nccl::send(exec, v, bad_rank, comm).wait();
  } else if (rank == dst) {
    KokkosComm::Experimental::nccl::recv(exec, v, bad_rank, comm).wait();
  }
}

}  // namespace
