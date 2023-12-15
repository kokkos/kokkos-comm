add_library(KokkosComm INTERFACE)

target_include_directories(KokkosComm INTERFACE impl)
target_link_libraries(KokkosComm INTERFACE MPI::MPI_CXX)