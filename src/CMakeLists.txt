add_library(KokkosComm INTERFACE)

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

file(GLOB_RECURSE KOKKOSCOMM_PUBLIC_HEADERS ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
set(KOKKOSCOMM_PUBLIC_HEADERS ${KOKKOSCOMM_PUBLIC_HEADERS} PARENT_SCOPE)