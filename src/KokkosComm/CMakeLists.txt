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
    FILE_SET HEADERS BASE_DIRS ${PROJECT_SOURCE_DIR}/src FILES impl/contiguous.hpp
    # Configuration header
    FILE_SET HEADERS BASE_DIRS ${CMAKE_BINARY_DIR}/src FILES ${CMAKE_BINARY_DIR}/src/KokkosComm/config.hpp
)

# Kokkos Comm is built on the ISO C++20 standard
target_compile_features(KokkosComm INTERFACE cxx_std_20)
set_target_properties(KokkosComm PROPERTIES CXX_EXTENSIONS OFF)

target_link_libraries(KokkosComm INTERFACE Kokkos::kokkos)

if(KokkosComm_ENABLE_MPI)
  add_subdirectory(mpi)
endif()
if(KokkosComm_ENABLE_NCCL)
  add_subdirectory(nccl)
endif()
