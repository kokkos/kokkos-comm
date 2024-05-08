#pragma once

class Communicator
{
private:
  MPI_Comm _raw_comm = MPI_COMM_WORLD;

public;
  Communicator(MPI_Comm mpi_communicator): _raw_comm{ mpi_communicator } {}
  operator MPI_Comm(){ return _raw_comm; }

  int size() const
  {
    int size;
    MPI_Comm_size( _raw_comm, &size );
    return size;
  }

  int rank() const
  {
    int rank;
    MPI_Comm_rank( _raw_comm, &rank );
    return rank;
  }

  void barrier() const { MPI_Barrier( _raw_comm ); }

  template< typename T, class... ARGS >
	void send( Kokkos::View< T const*, ARGS... > send_view, int dest_rank, int tag = 0 ) const
	{
		MPI_Send( send_view.data(), send_view.size(), mpi_type<T>(), dest_rank, tag, _raw_comm );
	}

  template< typename T, class... ARGS >
	void recv( Kokkos::View< T*, ARGS... > recv_view, int src_rank, int tag = 0 ) const
	{
		MPI_Recv( recv_view.data(), recv_view.size(), mpi_type<T>(), src_rank, tag, _raw_comm, MPI_STATUS_IGNORE );
	}

  template< typename T, class... ARGS >
	void sendrecv( Kokkos::View< T const*, ARGS... > send_view, Kokkos::View< T*, ARGS... > recv_view, int rank, int tag = 0 ) const
	{
		MPI_Sendrecv( send_view.data(), send_view.size(), mpi_type<T>(), rank, tag, _raw_comm,
						      recv_view.data(), recv_view.size(), mpi_type<T>(), rank, tag, _raw_comm, MPI_STATUS_IGNORE );
	}
};
