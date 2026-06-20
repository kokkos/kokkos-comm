// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstdio>
#include <cstdlib>

#include <fmt/core.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <hip/hip_runtime.h>
#include <stream-triggering.h>

//#include "../logging.hpp"

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

namespace test_utils::mpi_advance {

  template<class ExecutionSpace>
  class Ctx<ExecutionSpace>
  {
    using execution_space = ExecutionSpace;
  public:
  void* findStream( const ExecutionSpace& a )
    {
      if constexpr ( std::is_same_v<ExecutionSpace, Kokkos::HIP> )
        {
            // std::cout << "HIP STREAM!" << std::endl; // debug?
            return a.hip_stream();
        }
        else
        {
            // std::cout << "Base" << std::endl;
            return nullptr;
        }
    }
  //static auto init() -> Ctx {0
  Ctx(const ExecutionSpace& exec_space) {
    //hipStreamCreateWithFlags(&_my_stream, hipStreamNonBlocking)
    _my_stream = findStream(exec_space);
    MPIS_Queue_init( &_my_queue, CXI, &_my_stream );
    MPI_Info_create( &_mem_info );
    //return Ctx(); ; //Ctx(nccl_comm, stream, local_rank, n_ranks, my_rank);
  }

  // Forbid copies and moves
  Ctx(const Ctx &)                     = delete;
  auto operator=(const Ctx &) -> Ctx & = delete;
  Ctx(Ctx &&)                          = delete;
  auto operator=(Ctx &&) -> Ctx      & = delete;

  //auto comm() -> ncclComm_t & { return comm_; }
  //cudaStream_t stream() const { return stream_; }
  //auto size() -> int { return n_ranks_; }
  //auto rank() -> int { return my_rank_; }

  ~Ctx() {
    for(int i = 0; i < requests.size(); i++)
      {
	MPIS_Request_free(&requests[i]);
      }
    MPIS_Queue_free( &_my_queue );
    MPI_Info_free( &_mem_info );
  }
  
 private:
  //explicit Ctx(MPI_Comm comm)//, int dev, int n_ranks, int my_rank)
  //  : comm_(comm) {} // , stream_(stream), dev_(dev), n_ranks_(n_ranks), my_rank_(my_rank) {}

  void* _my_stream;
  //const MPI_Comm comm;
  MPIS_Queue _my_queue;
  MPI_Info _mem_info;
  std::vector<MPIS_Request> requests;

  //std::array<std::vector<MPIS_Request>, 2> _scatter_requests;
  //std::array<std::vector<MPIS_Request>, 2> _gather_requests;
  //std::vector<void*> _raw_buffers;
};

}  // namespace test_utils::nccl
