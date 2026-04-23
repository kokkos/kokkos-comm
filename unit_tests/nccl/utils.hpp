// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstdio>
#include <cstdlib>

#include <fmt/core.h>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

#include "../logging.hpp"

namespace {

[[nodiscard]] auto get_local_rank(MPI_Comm comm, int my_rank) -> int {
  MPI_Comm node_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &node_comm);

  int node_rank;
  MPI_Comm_rank(node_comm, &node_rank);

  MPI_Comm_free(&node_comm);
  return node_rank;
}

}  // namespace

namespace test_utils::nccl {

class Ctx {
 public:
  static auto init() -> Ctx {
    int flag;
    MPI_Initialized(&flag);
    KC_CHECK(flag == true, "MPI is not initialized");

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // Retrieve rank info
    int n_ranks, my_rank;
    MPI_Comm_size(mpi_comm, &n_ranks);
    MPI_Comm_rank(mpi_comm, &my_rank);
    int local_rank = get_local_rank(mpi_comm, my_rank);

    int n_gpus;
    KC_CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    KC_INFO("P{} found {} CUDA devices", my_rank, n_gpus);

    KC_CHECK(local_rank <= n_gpus, "P{} needs device #{} but only {} devices available", my_rank, local_rank, n_gpus);
    KC_CUDA_CHECK(cudaSetDevice(local_rank));
    KC_INFO("P{} assigned to CUDA device #{}", my_rank, local_rank);

    // Get NCCL unique ID at rank 0 and broadcast it to all others
    ncclUniqueId nccl_id;
    if (my_rank == 0) {
      ncclGetUniqueId(&nccl_id);
    }
    KC_MPI_CHECK(MPI_Bcast(&nccl_id, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, mpi_comm));

    // Initialize NCCL communicator
    ncclComm_t nccl_comm;
    KC_NCCL_CHECK(ncclCommInitRank(&nccl_comm, n_ranks, nccl_id, my_rank));

    cudaStream_t stream;
    KC_CUDA_CHECK(cudaStreamCreate(&stream));
    return Ctx(nccl_comm, stream, local_rank, n_ranks, my_rank);
  }

  ~Ctx() {
    KC_NCCL_CHECK(ncclCommDestroy(comm_));
    KC_CUDA_CHECK(cudaStreamDestroy(stream_));
  }
  // Forbid copies and moves
  Ctx(const Ctx &)                     = delete;
  auto operator=(const Ctx &) -> Ctx & = delete;
  Ctx(Ctx &&)                          = delete;
  auto operator=(Ctx &&) -> Ctx      & = delete;

  auto comm() -> ncclComm_t & { return comm_; }
  cudaStream_t stream() const { return stream_; }
  auto size() -> int { return n_ranks_; }
  auto rank() -> int { return my_rank_; }

 private:
  explicit Ctx(ncclComm_t comm, cudaStream_t stream, int dev, int n_ranks, int my_rank)
      : comm_(comm), stream_(stream), dev_(dev), n_ranks_(n_ranks), my_rank_(my_rank) {}

  ncclComm_t comm_;
  cudaStream_t stream_;  // my CUDA stream (associated with dev_)
  int dev_;              // my cuda device
  int n_ranks_;
  int my_rank_;
};

}  // namespace test_utils::nccl
