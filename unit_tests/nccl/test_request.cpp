// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <KokkosComm/KokkosComm.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <nccl.h>

#include <utility>

#ifndef ASSERT_CUDA
#define ASSERT_CUDA(call) ASSERT_EQ((call), cudaSuccess)
#endif
#ifndef ASSERT_NCCL
#define ASSERT_NCCL(call) ASSERT_EQ((call), ncclSuccess)
#endif

// Moving a Request must transfer ownership of the underlying handle:
// after the moved-from object is destroyed, the moved-to object must still
// hold a valid (non-null) handle.
TEST(Request, NcclMoveAssignKeepsHandleAfterSourceDestruction) {
  // Single-rank communicator: NCCL supports send/recv to self.
  ncclComm_t comm;
  int dev = 0;
  ASSERT_CUDA(cudaSetDevice(dev));
  ASSERT_NCCL(ncclCommInitAll(&comm, 1, &dev));

  cudaStream_t stream;
  ASSERT_CUDA(cudaStreamCreate(&stream));

  int *d_send = nullptr, *d_recv = nullptr;
  ASSERT_CUDA(cudaMalloc(&d_send, sizeof(int)));
  ASSERT_CUDA(cudaMalloc(&d_recv, sizeof(int)));
  const int send_value = 42;
  ASSERT_CUDA(cudaMemcpy(d_send, &send_value, sizeof(int), cudaMemcpyHostToDevice));
  ASSERT_CUDA(cudaMemset(d_recv, 0, sizeof(int)));

  // Explicit NCCL send/recv to self.
  ASSERT_NCCL(ncclGroupStart());
  ASSERT_NCCL(ncclSend(d_send, 1, ncclInt, 0, comm, stream));
  ASSERT_NCCL(ncclRecv(d_recv, 1, ncclInt, 0, comm, stream));
  ASSERT_NCCL(ncclGroupEnd());

  // ASSUMPTION: Request<NcclSpace> wraps a cudaEvent_t recorded on the stream.
  cudaEvent_t event = nullptr;
  ASSERT_CUDA(cudaEventCreate(&event));
  ASSERT_CUDA(cudaEventRecord(event, stream));

  KokkosComm::Request<KokkosComm::NcclSpace> persistent;
  {
    KokkosComm::Request<KokkosComm::NcclSpace> temporary(event);
    persistent = std::move(temporary);
  }  // temporary destroyed here

  // Only inspect the handle. Do not call wait().
  EXPECT_NE(persistent.request(), nullptr);

  // Clean up through the raw handles, not through `persistent`.
  ASSERT_CUDA(cudaStreamSynchronize(stream));
  int recv_value = -1;
  ASSERT_CUDA(cudaMemcpy(&recv_value, d_recv, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(recv_value, 42);

  ASSERT_CUDA(cudaEventDestroy(event));
  ASSERT_CUDA(cudaFree(d_send));
  ASSERT_CUDA(cudaFree(d_recv));
  ASSERT_CUDA(cudaStreamDestroy(stream));
  ASSERT_NCCL(ncclCommDestroy(comm));
}