target_sources(
  KokkosComm
  INTERFACE
    # Public MPI headers
    FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES
        # Objects
        mpi/mpi_space.hpp
        mpi/handle.hpp
        mpi/req.hpp
        mpi/comm_mode.hpp
        # Point-to-point
        mpi/send.hpp
        mpi/isend.hpp
        mpi/recv.hpp
        mpi/irecv.hpp
        mpi/channel.hpp
        # Collectives
        mpi/broadcast.hpp
        mpi/allgather.hpp
        mpi/alltoall.hpp
        mpi/allreduce.hpp
        mpi/reduce.hpp
        mpi/scan.hpp
        mpi/barrier.hpp
    # Implementation detail MPI headers
    FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES mpi/impl/pack_traits.hpp mpi/impl/packer.hpp mpi/impl/tags.hpp mpi/impl/error_handling.hpp
)

target_link_libraries(KokkosComm INTERFACE MPI::MPI_CXX)
