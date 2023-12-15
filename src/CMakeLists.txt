add_library(KokkosComm INTERFACE)

target_include_directories(KokkosComm INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(KokkosComm INTERFACE ${CMAKE_CURRENT_LIST_DIR}/impl)
target_link_libraries(KokkosComm INTERFACE MPI::MPI_CXX Kokkos::kokkos)