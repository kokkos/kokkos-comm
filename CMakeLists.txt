# 3.9: MPI::MPI_CXX
cmake_minimum_required(VERSION 3.9)
project(KokkosComm)

find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

message(STATUS "MPI_VERSION =                 ${MPI_VERSION}")
message(STATUS "MPI_CXX_COMPILER =            ${MPI_CXX_COMPILER}")
message(STATUS "MPI_CXX_COMPILE_OPTIONS =     ${MPI_CXX_COMPILE_OPTIONS}")
message(STATUS "MPI_CXX_COMPILE_DEFINITIONS = ${MPI_CXX_COMPILE_DEFINITIONS}")
message(STATUS "MPI_CXX_INCLUDE_DIRS =        ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI_CXX_LINK_FLAGS =          ${MPI_CXX_LINK_FLAGS}")
message(STATUS "MPI_CXX_LIBRARIES =           ${MPI_CXX_LIBRARIES}")
add_subdirectory(src)

enable_testing()
add_subdirectory(unit_tests)