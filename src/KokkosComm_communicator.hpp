#pragma once

#include <vector>
#include <optional>
#include <memory>
#include <Kokkos_Core.hpp>
#include "KokkosComm_concepts.hpp"

namespace KokkosComm {
  // Basic encapsultation type for MPI_Request (prevents implicit conversions + more user-friendly API)
  class Request {
   private:
    MPI_Request _raw_req;
  
   public:
    Request( MPI_Request request ): _raw_req{ request } {}
    operator MPI_Request() const { return _raw_req; }

    void wait(){ MPI_Wait( &_raw_req, MPI_STATUS_IGNORE ); }
    void free(){ MPI_Request_free( &_raw_req ); }
  };

  // Simple encapsultation type for MPI_Comm (same as above)
  class Communicator
  {
   private:
    MPI_Comm _raw_comm = MPI_COMM_WORLD;
    bool _blocking_semantics = false;
    CommMode _default_mode = CommMode::Default;

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
    std::optional<Request> send( Kokkos::View< T const*, ARGS... > send_view, int dest_rank, int tag = 0 ) const {
      if( _blocking_semantics ){
        switch ( mode ){
          case CommMode::Standard:
            MPI_Send( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm );
	    return std::nullopt;
          case CommMode::Ready:
            MPI_Rsend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm );
	    return std::nullopt;
          case CommMode::Synchronous:
            MPI_Ssend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm );
	    return std::nullopt;
          case CommMode::Default:
            #ifdef KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE
            MPI_Ssend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm );
            #else
            MPI_Send( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm );
            #endif
	    return std::nullopt;
        }
      }
      else {
	MPI_Request req;
        switch ( mode ){
          case CommMode::Standard:
            MPI_Isend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm, &req ); 
	    return { req };
          case CommMode::Ready:
            MPI_Irsend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm, &req ); 
	    return { req };
          case CommMode::Synchronous:
            MPI_Issend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm, &req );
	    return { req };
          case CommMode::Default:
            #ifdef KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE
            MPI_Issend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm, &req );
            #else
            MPI_Isend( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm, &req ); 
            #endif
	    return { req };
        }
      }
      throw std::runtime_error{ "Unknown communication mode" };
    }


    template< typename T, class... ARGS >
    std::optional<Request> recv( Kokkos::View< T*, ARGS... > recv_view, int src_rank, int tag = 0 ) const {
      if( _blocking_semantics ){
        MPI_Recv( recv_view.data(), recv_view.size(), mpi_type<T>(), src_rank, tag, _raw_comm, MPI_STATUS_IGNORE );
	return std::nullopt;
      else {
        MPI_Request req;
        MPI_Irecv( recv_view.data(), recv_view.size(), mpi_type<T>(), src_rank, tag, _raw_comm, &req );
        return { req };
      }
    }
  };


  // A dynamic scheme similar to what is currently implemented, with packing and unpacking on the fly
  class TypeErasedScheme
  {
   private:
    struct ViewConcept {
      virtual ~ViewConcept() = 0;
    };

    template< typename View >
    struct ViewModel : public ViewConcept {
      View _view;
    };
    
    struct SendRecord {
      Request request;
      std::unique_ptr<ViewConcept> view;
    };

    struct RecvRecord {
      Request request;
      std::unique_ptr<ViewConcept> view;
      std::optional< std::unique_ptr<ViewConcept> > unpack_view;
    };

    std::shared_ptr< std::vector<SendRecord> > _send_records;
    std::shared_ptr< std::vector<RecvRecord> > _recv_records;
    Communicator _comm;

   public:
    template< typename ExecSpace, typename View >
    void send( ExecSpace space, View view, int dest_rank, int tag = 0 ){
      using T = typename View::value_type;
      using ViewTraits = KokkosComm::Traits<View>;

      if ( KokkosComm::PackTraits<View>::needs_pack(view) ) {
        using Packer  = typename KokkosComm::PackTraits<View>::packer_type;
        using MpiArgs = typename Packer::args_type;

        MpiArgs args = Packer::pack(space, view);
        space.fence();
        auto req = comm.send( KokkosComm::Traits<View>::data_handle(args.view), args.count, args.datatype, dest_rank, tag );
        _send_records->push_back({ req, std::make_unique< ViewModel<decltype(args.view)> >(args.view) });
      } 
      else {
        auto req = comm.send( ViewTraits::data_handle(view), ViewTraits::span(view), mpi_type<T>(), dest_rank, tag );
        _send_records->push_back({ req, std::make_unique< ViewModel<View> >(view) });
      }
    }

    template< typename ExecSpace, typename View >
    void recv( ExecSpace space, View view, int src_rank, int tag = 0 ){
      using T = typename View::value_type;
      using ViewTraits = KokkosComm::Traits<View>;

      if ( KokkosComm::PackTraits<View>::needs_pack(view) ) {
        using Packer  = typename KokkosComm::PackTraits<View>::packer_type;
        using MpiArgs = typename Packer::args_type;

        auto args = Packer::allocate_packed_for( space, "", view );
        auto req = comm.recv( KokkosComm::Traits<View>::data_handle(args.view), args.count, args.datatype, src_rank, tag );
        _recv_records->push_back({ req, std::make_unique< ViewModel<decltype(args.view)> >(args.view), std::make_unique< ViewModel<View> >(view) });
      } 
      else {
        auto req = comm.recv( ViewTraits::data_handle(view), ViewTraits::span(view), mpi_type<T>(), src_rank, tag );
        _recv_records->push_back({ req, std::make_unique< ViewModel<View> >(view), std::nullopt });
      }
    }

    void wait(){
      while( _recv_records->size() > 0 ){
        _recv_records->back().request.wait(); // it looks like more TE is needed to make this work...
        // if ( _recv_records->back().unpack_view ){
        //   Packer::unpack_into( space, _recv_records->back().unpack_view, _recv_records->back().view );
        //   space.fence();
        // }
        _recv_records->pop_back();
      }
      while( _send_records->size() > 0 ){
        _send_records->back().request.wait();
        _send_records->pop_back();
      }
    }
  };



  // Another example: a simple static communication scheme which gathers inputs with a callback fcn into 
  // a single send buffer, then sends and receives asynchronously, and directly exposes the values from 
  // the receive buffer to minimize allocations and copies
  template< typename T, typename ExecSpace >
  class BufferedScheme
  {
   private:
    using SendView = Kokkos::View< T const*, Kokkos::LayoutStride, Kokkos::MemoryTraits< Kokkos::Unmanaged > >;
    using RecvView = Kokkos::View< T*, Kokkos::LayoutStride, Kokkos::MemoryTraits< Kokkos::Unmanaged > >;

    Communicator _comm;

    // Send config
    std::vector< int > _dest_ranks;
    std::vector< int > _send_offsets;
    Kokkos::View< T* > _send_buffer;

    // Recv config
    std::vector< int > _src_ranks;
    std::vector< int > _recv_offsets;
    Kokkos::View< T* > _recv_buffer;
 
    std::vector< Request > _recv_requests;

   public:
    template< typename Callable >
    void launch( Callable pack_fcn ){
      int offset = 0;
      Kokkos::parallel_for(
        "KokkosComm::BufferedScheme::launch"
        Kokkos::RangePolicy<ExecSpace>{ 0, _dest_ranks.size() },
          KOKKOS_LAMBDA( int dest ){
          for ( int i = _send_offsets(dest); i < _send_offsets(dest+1); ++i ){
            _send_buffer(i) = pack_fcn(dest, i-_send_offsets(dest));
          }
        }
      );
      
      offset = 0;
      for ( auto& [ src, view, req ] : std::ranges::views::zip( _src_ranks, _recv_views, _recv_requests ) ){
        req = comm.recv( Kokkos::subview( _recv_buffer, Kokkos::pair{ offset, offset+view.size } ), src );
        offset += view.size();
      }

      offset = 0;
      for ( auto& [ dest, view ] : std::ranges::views::zip( _dest_ranks, _send_views ) ){
        comm.send( Kokkos::subview( _send_buffer, Kokkos::pair{ offset, offset+view.size } ), dest ).free();
        offset += view.size();
      }
    }

    void wait(){ MPI_Waitall( _recv_requests.size(), _recv_requests.data(), MPI_STATUSES_IGNORE ); }

    T operator()( int src, int idx ) const { return _recv_buffer( _recv_offsets( src ) + idx ); }
  };
}
