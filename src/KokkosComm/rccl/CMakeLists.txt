target_sources(
  KokkosComm
  INTERFACE
    # Public RCCL headers
    FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES
        # Objects
        rccl_space.hpp
        communicator.hpp
        request.hpp
        # Point-to-point
        send.hpp
        recv.hpp
        # Collectives
        broadcast.hpp
        allgather.hpp
        alltoall.hpp
        allreduce.hpp
        reduce.hpp
    # Implementation detail RCCL headers
    FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES impl/pack_traits.hpp impl/packer.hpp impl/error_handling.hpp
)

target_link_libraries(KokkosComm INTERFACE rccl hip::host)
