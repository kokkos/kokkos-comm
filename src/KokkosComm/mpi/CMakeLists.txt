target_sources(
  KokkosComm
  INTERFACE
    # Public MPI headers
    FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES
        # Objects
        mpi_space.hpp
        communicator.hpp
        request.hpp
        comm_mode.hpp
        # Point-to-point
        send.hpp
        isend.hpp
        recv.hpp
        irecv.hpp
        channel.hpp
        # Collectives
        broadcast.hpp
        allgather.hpp
        alltoall.hpp
        allreduce.hpp
        reduce.hpp
        scan.hpp
        barrier.hpp
    # Implementation detail MPI headers
    FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES impl/pack_traits.hpp impl/packer.hpp impl/tags.hpp impl/error_handling.hpp
)

target_link_libraries(KokkosComm INTERFACE MPI::MPI_CXX)
