add_library(KokkosComm INTERFACE)

target_include_directories(KokkosComm INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(KokkosComm INTERFACE MPI::MPI_CXX Kokkos::kokkos)
if(KOKKOSCOMM_USE_KOKKOS_MDSPAN)
  target_link_libraries(KokkosComm INTERFACE mdspan)
endif()

kokkoscomm_add_cxx_flags(TARGET KokkosComm INTERFACE)

# FIXME Not a good idea, we should explicitely list the headers
file(GLOB_RECURSE KOKKOSCOMM_PUBLIC_HEADERS ${CMAKE_CURRENT_LIST_DIR}/*.hpp)

set_target_properties(KokkosComm PROPERTIES PUBLIC_HEADER "${KOKKOSCOMM_PUBLIC_HEADERS}")
