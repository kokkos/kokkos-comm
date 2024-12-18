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

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA_CHECK(cmd)                            \
  if (cudaError_t err = cmd; err != cudaSuccess) { \
    std::cerr << cudaGetErrorString(err) << "\n";  \
    exit(1);                                       \
  }

#define NCCL_CHECK(cmd)                             \
  if (ncclResult_t res = cmd; res != ncclSuccess) { \
    std::cerr << ncclGetErrorString(res) << "\n";   \
    exit(2);                                        \
  }

std::string uid_to_string(const ncclUniqueId &id) {
  std::stringstream ss;
  ss << std::hex << std::setfill('0');
  for (size_t i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
    ss << std::setw(2) << int(((char *)id.internal)[i]);
  }
  return ss.str();
}

int main(int argc, char *argv[]) {
  int nDevices;
  CUDA_CHECK(cudaGetDeviceCount(&nDevices));
  std::cout << "Found " << nDevices << " GPUs" << std::endl;

  // Initialize NCCL communicators
  ncclUniqueId id;
  NCCL_CHECK(ncclGetUniqueId(&id));
  std::cout << uid_to_string(id) << std::endl;

  std::vector<float *> sendbuff(nDevices);
  std::vector<float *> recvbuff(nDevices);
  std::vector<cudaStream_t> streams(nDevices);
  std::vector<ncclComm_t> comms(nDevices);
  const size_t size = 4;  // 4 entries per GPU

  // Initialize each GPU and allocate memory
  for (int i = 0; i < nDevices; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaMalloc(&sendbuff[i], size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recvbuff[i], size * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // Initialize data on each GPU
    // GPU 0:  0  1  2  3
    // GPU 1:  4  5  6  7
    // GPU 2:  8  9 10 11
    // GPU 3: 12 13 14 15
    // ...
    std::vector<float> hostData(size);
    for (int j = 0; j < size; ++j) {
      hostData[j] = i * size + j;  // Different data on each GPU
    }
    CUDA_CHECK(cudaMemcpy(sendbuff[i], hostData.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    NCCL_CHECK(ncclCommInitRank(&comms[i], nDevices, id, i));
  }

  // Perform all-reduce operation
  NCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < nDevices; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    NCCL_CHECK(
        ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], size, ncclFloat, ncclSum, comms[i], streams[i]));
  }
  NCCL_CHECK(ncclGroupEnd());

  // Synchronize and verify results
  // 1 GPU total =  0  1  2  3
  // 2 GPU total =  4  6  8 10
  // 3 GPU total = 12 15 18 21
  // 4 GPU total = 24 28 32 36
  std::vector<float> results(size);
  for (int i = 0; i < nDevices; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));

    CUDA_CHECK(cudaMemcpy(results.data(), recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost));

    for (int j = 0; j < size; ++j) {
      int expected = (nDevices) * (nDevices - 1) * size + nDevices * j;
      if (results[j] != expected) {
        std::cerr << "error on device " << i << " @ " << j << " expected=" << expected << " actual=" << results[j]
                  << "\n";
      }
    }
  }

  // Cleanup
  for (int i = 0; i < nDevices; ++i) {
    NCCL_CHECK(ncclCommDestroy(comms[i]));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaFree(sendbuff[i]));
    CUDA_CHECK(cudaFree(recvbuff[i]));
  }

  return 0;
}