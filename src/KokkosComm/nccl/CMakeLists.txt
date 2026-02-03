target_sources(
  KokkosComm
  INTERFACE
    # Public NCCL headers
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
        nccl/allgather.hpp
        nccl/alltoall.hpp
        nccl/allreduce.hpp
        nccl/reduce.hpp
    # Implementation detail NCCL headers
    FILE_SET HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES nccl/impl/pack_traits.hpp nccl/impl/packer.hpp nccl/impl/types.hpp
)

target_link_libraries(KokkosComm INTERFACE NCCL::NCCL)
