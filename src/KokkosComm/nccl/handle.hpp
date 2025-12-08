// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cuda.h>
#include <nccl.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/fwd.hpp>
#include "nccl_space.hpp"

namespace KokkosComm {

template <>
class Handle<Kokkos::Cuda, Experimental::NcclSpace> {
 public:
  using execution_space     = Kokkos::Cuda;
  using communication_space = Experimental::NcclSpace;
  using handle_type         = communication_space::handle_type;
  using datatype_type       = communication_space::datatype_type;
  using reduction_op_type   = communication_space::reduction_op_type;
  using rank_type           = communication_space::rank_type;

  explicit Handle(const execution_space& space, handle_type comm) : space_(space), comm_(comm) {}
  explicit Handle(handle_type comm) : Handle(execution_space{}, comm) {}

  /// This initializes NCCL communicators for multiple GPUs within a single process.
  /// It is the simplest NCCL setup, as it does not rely on MPI, and is ideal for applications that want to use
  /// multiple GPUs without the complexity of multi-process coordination.
  ///
  /// FIXME: Do we want to allow users creating a NCCL Handle without providing the communicator?
  /// This requires us initializing it manually, which, if we do distributed initialization, is a lot more work than
  /// for initializing MPI. The implementation below does not really match what KokkosComm is trying to solve,
  /// i.e. enabling multi-node communications.
  /// Commenting it out for now.
  // Handle() {
  //   // Discover how many CUDA devices are available
  //   int n_gpus;
  //   cudaGetDeviceCount(&n_gpus);
  //   if (n_gpus == 0) {
  //     std::cerr << "KokkosComm::Handle<Cuda, Nccl>: error: no CUDA devices found\n";
  //     Kokkos::abort();
  //   }

  //   // Allocate arrays to hold our per-device resources
  //   // We need one communicator, stream, and device ID per GPU
  //   std::vector<int> dev_ids(n_gpus);
  //   std::vector<ncclComm_t> comms(n_gpus);
  //   std::vector<cudaStream_t> streams(n_gpus);

  //   // Create device list. Default: use all available devices.
  //   for (int i = 0; i < n_gpus; ++i) {
  //     dev_ids[i] = i;  // Use device i for communicator i
  //     cudaSetDevice(dev_ids[i]);
  //     cudaStreamCreate(&streams[i]);
  //   }

  //   // Initialize NCCL communicators.
  //   ncclCommInitAll(comms, n_gpus, dev_ids);
  //   // This is semantically wrong, since our Handle only remembers the communicator of the first device.
  //   return Handle(execution_space{}, comms[0]);
  // }

  auto comm() -> handle_type& { return comm_; }
  auto comm() const -> const handle_type& { return comm_; }
  auto space() -> execution_space& { return space_; }
  auto space() const -> const execution_space& { return space_; }

  auto size() -> rank_type {
    rank_type ret;
    ncclCommCount(comm_, &ret);
    return ret;
  }

  auto rank() -> rank_type {
    rank_type ret;
    ncclCommUserRank(comm_, &ret);
    return ret;
  }

 private:
  execution_space space_;
  handle_type comm_;
};

}  // namespace KokkosComm
