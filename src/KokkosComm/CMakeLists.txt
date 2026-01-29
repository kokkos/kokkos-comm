# Public headers
target_sources(
  KokkosComm
  INTERFACE
    FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES
        # Public headers
        KokkosComm.hpp
        collective.hpp
        concepts.hpp
        fwd.hpp
        point_to_point.hpp
        traits.hpp
        datatype.hpp
        reduction_op.hpp
        # Implementation detail headers
        impl/contiguous.hpp
    FILE_SET HEADERS
      BASE_DIRS ${CMAKE_BINARY_DIR}/src
      FILES ${CMAKE_BINARY_DIR}/src/KokkosComm/config.hpp
)

if(KokkosComm_ENABLE_MPI)
  target_sources(
    KokkosComm
    INTERFACE
      FILE_SET HEADERS
        BASE_DIRS ${PROJECT_SOURCE_DIR}/src
        FILES
          # Public MPI headers
          # Objects
          mpi/mpi_space.hpp
          mpi/comm_mode.hpp
          mpi/handle.hpp
          mpi/req.hpp
          mpi/comm_mode.hpp
          mpi/channel.hpp
          # P2P
          mpi/irecv.hpp
          mpi/isend.hpp
          mpi/recv.hpp
          mpi/send.hpp
          # Collectives
          mpi/broadcast.hpp
          mpi/allgather.hpp
          mpi/allreduce.hpp
          mpi/alltoall.hpp
          mpi/reduce.hpp
          mpi/scan.hpp
          # Other/utilities
          mpi/barrier.hpp
          # Implementation detail MPI headers
          mpi/impl/pack_traits.hpp
          mpi/impl/packer.hpp
          mpi/impl/tags.hpp
          mpi/impl/error_handling.hpp
  )
endif()

if(KokkosComm_ENABLE_NCCL)
  target_sources(
    KokkosComm
    INTERFACE
      FILE_SET HEADERS
        BASE_DIRS ${PROJECT_SOURCE_DIR}/src
        FILES
          # Public NCCL headers
          # Structures
          nccl/nccl_space.hpp
          nccl/handle.hpp
          nccl/req.hpp
          # P2P
          nccl/send.hpp
          nccl/recv.hpp
          # Collectives
          nccl/broadcast.hpp
          nccl/alltoall.hpp
          nccl/allgather.hpp
          nccl/allreduce.hpp
          nccl/reduce.hpp
          # Implementation detail NCCL headers
          nccl/impl/pack_traits.hpp
          nccl/impl/packer.hpp
          nccl/impl/types.hpp
  )
endif()

# Kokkos Comm is built on the ISO C++20 standard
target_compile_features(KokkosComm INTERFACE cxx_std_20)
set_target_properties(
  KokkosComm
  PROPERTIES
    CXX_EXTENSIONS
      OFF
)

# Linking
target_link_libraries(KokkosComm INTERFACE Kokkos::kokkos)
if(KOKKOSCOMM_ENABLE_MPI)
  target_link_libraries(KokkosComm INTERFACE MPI::MPI_CXX)
endif()
if(KOKKOSCOMM_ENABLE_NCCL)
  target_link_libraries(KokkosComm INTERFACE NCCL::NCCL)
endif()
