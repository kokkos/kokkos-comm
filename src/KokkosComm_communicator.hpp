#pragma once

#include <vector>
#include <Kokkos_Core.hpp>
#include "KokkosComm_concepts.hpp"

namespace KokkosComm {
  class Request {
   private:
    MPI_Request _raw_req;
  
   public:
    Request( MPI_Request request ): _raw_req{ request } {}
    operator MPI_Request() const { return _raw_req; }

    void wait(){ MPI_Wait( &_raw_req, MPI_STATUS_IGNORE ); }
		void free(){ MPI_Request_free( &_raw_req ); }
  };

  class Communicator
  {
   private:
    MPI_Comm _raw_comm = MPI_COMM_WORLD;

   public:
    Communicator(MPI_Comm mpi_communicator): _raw_comm{ mpi_communicator } {}
    operator MPI_Comm(){ return _raw_comm; }

    int size() const {
      int size;
      MPI_Comm_size( _raw_comm, &size );
      return size;
    }

    int rank() const {
      int rank;
      MPI_Comm_rank( _raw_comm, &rank );
      return rank;
    }

    void barrier() const { MPI_Barrier( _raw_comm ); }

    template< typename T, class... ARGS >
    void send( Kokkos::View< T const*, ARGS... > send_view, int dest_rank, int tag = 0 ) const {
      switch ( mode ){
        case CommMode::Standard:
          MPI_Send( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm );
          break;
        case CommMode::Ready:
          MPI_Rsend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm );
          break;
        case CommMode::Synchronous:
          MPI_Ssend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm );
          break;
        case CommMode::Default:
          #ifdef KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE
          MPI_Ssend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm );
          #else
          MPI_Send( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm );
          #endif
        default:
          throw std::runtime_error{ "Unknown communication mode" };
      }
    }

    template< typename T, class... ARGS >
    KokkosComm::Request isend( Kokkos::View< T const*, ARGS... > const& send_view, int dest_rank, int tag = 0, CommMode mode = CommMode::Default ){
      MPI_Request req;
      switch ( mode ){
        case CommMode::Standard:
          MPI_Isend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm, &req );
          break;
        case CommMode::Ready:
          MPI_Irsend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm, &req );
          break;
        case CommMode::Synchronous:
          MPI_Issend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm, &req);
          break;
        case CommMode::Default:
          #ifdef KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE
          MPI_Issend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm, &req );
          #else
          MPI_Isend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm, &req );
          #endif
        default:
          throw std::runtime_error{ "Unknown communication mode" };
      }
      return KokkosComm::Request{ req };
    }


    template< typename T, class... ARGS >
    void recv( Kokkos::View< T*, ARGS... > recv_view, int src_rank, int tag = 0 ) const {
      MPI_Recv( recv_view.data(), recv_view.size(), mpi_type<T>(), src_rank, tag, _raw_comm, MPI_STATUS_IGNORE );
    }


    template< typename T, class... ARGS >
    void sendrecv( Kokkos::View< T const*, ARGS... > send_view, Kokkos::View< T*, ARGS... > recv_view, int rank, int tag = 0 ) const {
      MPI_Sendrecv( send_view.data(), send_view.size(), mpi_type<T>(), rank, tag, _raw_comm,
                    recv_view.data(), recv_view.size(), mpi_type<T>(), rank, tag, _raw_comm, MPI_STATUS_IGNORE );
    }
  };

  template< typename T >
  concept CommScheme = requires ( T cs, Communicator comm ){
    cs = T{ comm };
    cs.launch();
    cs.wait();
  };

  template< typename T >
  class BufferedScheme
  {
   private:
    using SendView = Kokkos::View< T const*, Kokkos::LayoutStride, Kokkos::MemoryTraits< Kokkos::Unmanaged > >;
    using RecvView = Kokkos::View< T*, Kokkos::LayoutStride, Kokkos::MemoryTraits< Kokkos::Unmanaged > >;

    Communicator _comm;

    std::vector< int > _src_ranks;
    std::vector< RecvView > _recv_views;
    bool _recv_updated;
   
    std::vector< int > _dest_ranks;
    std::vector< SendView > _send_views;
    bool _send_updated;
 
    Kokkos::View< T* > _recv_buffer;
    Kokkos::View< T* > _send_buffer;
 
    std::vector< Request > _recv_requests;
    std::future< void > _wait_handle;

   public:
    void launch(){
      _wait_handle = std::async( std::launch::async, [&]{
        if ( _recv_updated )
          Kokkos::resize( _recv_buffer, std::accumulate( ... ) );
        
        int offset = 0;
        for ( auto& [ src, view, req ] : std::ranges::views::zip( _src_ranks, _recv_views, _recv_requests ) ){
          req = comm.irecv( Kokkos::subview( _recv_buffer, Kokkos::pair{ offset, offset+view.size } ), src );
          offset += view.size();
        }


        if ( _send_updated )
          Kokkos::resize( _send_buffer, std::accumulate( ... ) );

        // pack

        offset = 0;
        for ( auto& [ dest, view ] : std::ranges::views::zip( _dest_ranks, _send_views ) ){
          comm.isend( Kokkos::subview( _send_buffer, Kokkos::pair{ offset, offset+view.size } ), dest ).free();
          offset += view.size();
        }


        MPI_Waitall( _recv_requests.size(), _recv_requests.data(), MPI_STATUSES_IGNORE ); 
        
        // unpack
      });
    }

    void wait(){ _wait_handle.wait(); }
  };
}