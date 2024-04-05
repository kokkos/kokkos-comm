add_library(KokkosComm INTERFACE)
add_library(KokkosComm::KokkosComm ALIAS KokkosComm)

target_include_directories(KokkosComm INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)
target_include_directories(KokkosComm INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/impl>
)
target_include_directories(KokkosComm INTERFACE
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(KokkosComm INTERFACE MPI::MPI_CXX Kokkos::kokkos)
if(KOKKOSCOMM_USE_KOKKOS_MDSPAN)
  target_link_libraries(KokkosComm INTERFACE mdspan)
endif()

file(GLOB_RECURSE KOKKOSCOMM_PUBLIC_HEADERS ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
set(KOKKOSCOMM_PUBLIC_HEADERS ${KOKKOSCOMM_PUBLIC_HEADERS} PARENT_SCOPE)