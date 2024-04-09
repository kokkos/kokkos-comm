add_library(KokkosComm INTERFACE)
add_library(KokkosComm::KokkosComm ALIAS KokkosComm)

target_include_directories(KokkosComm INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)
target_include_directories(KokkosComm INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/impl>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(KokkosComm INTERFACE MPI::MPI_CXX Kokkos::kokkos)

target_link_libraries(KokkosComm INTERFACE KokkosComm::KokkosCommFlags)

file(GLOB_RECURSE KOKKOSCOMM_PUBLIC_HEADERS ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
set(KOKKOSCOMM_PUBLIC_HEADERS ${KOKKOSCOMM_PUBLIC_HEADERS} PARENT_SCOPE)
