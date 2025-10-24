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

# KokkosComm is a C++20 project
target_compile_features(KokkosComm INTERFACE cxx_std_20)
set_target_properties(
  KokkosComm
  PROPERTIES
    CXX_EXTENSIONS
      OFF
)

# Definitions
if(KokkosComm_ENABLE_MPI)
  target_compile_definitions(KokkosComm INTERFACE KOKKOSCOMM_ENABLE_MPI)
  if(KokkosComm_IMPL_MPI_IS_MPICH)
    target_compile_definitions(KokkosComm INTERFACE KOKKOSCOMM_IMPL_MPI_IS_MPICH)
  elseif(KokkosComm_IMPL_MPI_IS_OPENMPI)
    target_compile_definitions(KokkosComm INTERFACE KOKKOSCOMM_IMPL_MPI_IS_OPENMPI)
  endif()
  if(KokkosComm_IMPL_MPI_HAS_MPI_EXT_H)
    target_compile_definitions(KokkosComm INTERFACE KOKKOSCOMM_IMPL_MPIEXT_H)
  endif()
endif()
if(KokkosComm_ENABLE_NCCL)
  target_compile_definitions(KokkosComm INTERFACE KOKKOSCOMM_ENABLE_NCCL)
endif()
if(KokkosComm_ENABLE_ABORT_ON_ERROR)
  target_compile_definitions(KokkosComm INTERFACE KOKKOSCOMM_ABORT_ON_ERROR)
endif()

# Linking
target_link_libraries(KokkosComm INTERFACE Kokkos::kokkos)
if(KokkosComm_ENABLE_MPI)
  target_link_libraries(KokkosComm INTERFACE MPI::MPI_CXX)
endif()
if(KokkosComm_ENABLE_NCCL)
  target_link_libraries(KokkosComm INTERFACE NCCL::NCCL)
endif()
