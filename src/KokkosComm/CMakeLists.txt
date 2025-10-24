target_sources(
  KokkosComm
  INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      # Public headers
      KokkosComm.hpp
      fwd.hpp
      concepts.hpp
      traits.hpp
      collective.hpp
      point_to_point.hpp
      reduction_op.hpp
      # Implementation detail headers
      impl/contiguous.hpp
)
# Configuration header
target_sources(
  KokkosComm
  INTERFACE
    FILE_SET kokkoscomm_config_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_BINARY_DIR}/src
    FILES ${PROJECT_BINARY_DIR}/src/KokkosComm/config.hpp
)

if(KokkosComm_ENABLE_MPI)
  target_sources(
    KokkosComm
    INTERFACE
      FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES
        # Objects
        mpi/mpi_space.hpp
        mpi/handle.hpp
        mpi/req.hpp
        mpi/comm_mode.hpp
        mpi/channel.hpp
        # Point-to-point
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
        # Implementation detail headers
        mpi/impl/pack_traits.hpp
        mpi/impl/packer.hpp
        mpi/impl/tags.hpp
        mpi/impl/types.hpp
        mpi/impl/error_handling.hpp
  )
endif()
# Should this be an `elseif()`?
if(KokkosComm_ENABLE_NCCL)
  target_sources(
    KokkosComm
    INTERFACE
      FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES
        # Objects
        nccl/nccl_space.hpp
        nccl/handle.hpp
        nccl/req.hpp
        # Point-to-point
        nccl/send.hpp
        nccl/recv.hpp
        # Collectives
        nccl/broadcast.hpp
        nccl/alltoall.hpp
        nccl/allgather.hpp
        nccl/allreduce.hpp
        nccl/reduce.hpp
        # Implementation detail headers
        nccl/impl/pack_traits.hpp
        nccl/impl/packer.hpp
        nccl/impl/types.hpp
  )
endif()

if(KokkosComm_ABORT_ON_ERROR)
  target_compile_definitions(KokkosComm INTERFACE KOKKOSCOMM_ABORT_ON_ERROR)
endif()

# KokkosComm is a C++20 project
target_compile_features(KokkosComm INTERFACE cxx_std_20)
set_target_properties(
  KokkosComm
  PROPERTIES
    CXX_EXTENSIONS
      OFF
)


# Linking
target_link_libraries(KokkosComm INTERFACE Kokkos::kokkos)
if(KokkosComm_ENABLE_MPI)
  target_link_libraries(KokkosComm INTERFACE MPI::MPI_CXX)
endif()
if(KokkosComm_ENABLE_NCCL)
  target_link_libraries(KokkosComm INTERFACE NCCL::NCCL)
endif()
# Install library
install(
  TARGETS
    KokkosComm
    KokkosCommFlags
  EXPORT KokkosCommTargets
  FILE_SET
  kokkoscomm_public_headers
  FILE_SET
  kokkoscomm_impl_headers
  FILE_SET
  kokkoscomm_mpi_headers
  FILE_SET
  kokkoscomm_mpi_impl_headers
  FILE_SET
  kokkoscomm_nccl_headers
  FILE_SET
  kokkoscomm_nccl_impl_headers
  FILE_SET
  kokkoscomm_config_headers
)
