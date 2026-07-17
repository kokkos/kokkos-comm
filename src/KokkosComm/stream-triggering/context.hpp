// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <stream-triggering.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "comm_mode.hpp"

#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm::Experimental {
namespace stream {
  
  //template <class ExecutionSpace>
  class StreamContext//<ExecutionSpace>
  {
    //using execution_space = ExecutionSpace;
  public:

    MPI_Info get_mem_info() const{
      return this->_mem_info;
    }

    MPIS_Queue get_queue() const{
      return this->_my_queue;
    }
    
    void add_request(const MPIS_Request req)
    {
      requests.push_back(req);
    }

    void block(){
      MPIS_Enqueue_startall( _my_queue, requests.size(), &requests[0]);
      MPIS_Enqueue_waitall( _my_queue);
    }

    void block(int num_of_reqs, MPIS_Request reqs[]){
      MPIS_Matchall(num_of_reqs, &reqs[0], MPI_STATUS_IGNORE);
      MPIS_Enqueue_startall( _my_queue, num_of_reqs, &reqs[0]);
      MPIS_Enqueue_waitall( _my_queue);
    }
    
    StreamContext()//const ExecutionSpace& exec_space)
    {
      // possible switch between fine and coarse
      if ( const char* env_db =
	   std::getenv( "MPI_ADVANCE_DOUBLE_BUFFERING" ) )
	{
	  _double_buffer = atoi( env_db );
	}
      // if no env variable is found, set to zero
      else
	{
	  _double_buffer = 1;
	}
      // set up fine grain memory
      if ( const char* env_fg =
	   std::getenv( "MPI_ADVANCE_FINEGRAIN_MEMORY" ) )
	{
	  _fine_grain = atoi( env_fg );
	}
      // if no env variable is found, set to zero
      else
	{
	  _fine_grain = 0;
	}
      MPI_Info_create( &_mem_info );
      if ( _fine_grain )
        {
	  MPI_Info_set( _mem_info, "mpi_memory_alloc_kinds",
			"rocm:device:fine" );
        }
      else
        {
	  MPI_Info_set( _mem_info, "mpi_memory_alloc_kinds",
			"rocm:device:coarse" );
        }

      if constexpr ( std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::HIP>){
		 _my_stream = Kokkos::HIP().hip_stream();
  }
      else{
	_my_stream = nullptr;
      }
      MPIS_Queue_init(&_my_queue, CXI, &_my_stream);
      
    }
    
    ~StreamContext()
    {
      for(int i = 0; i < requests.size(); i++){
	MPIS_Request_free(&requests[i]);
      }
      MPIS_Queue_free( &_my_queue );
      MPI_Info_free( &_mem_info );
    }
    
  private:
    void* _my_stream;
    MPI_Info _mem_info;
    MPIS_Queue _my_queue;
    std::vector<MPIS_Request> requests;
    int _double_buffer, _fine_grain;
    //std::vector<Impl::Packer::args_type> packed_buffers;
  };
  
}  // namespace stream
}  // namespace KokkosComm
