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
};
