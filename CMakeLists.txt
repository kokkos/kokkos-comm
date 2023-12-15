# 3.9: MPI::MPI_CXX
# 3.12: CMAKE_PROJECT_VERSION_MAJOR
cmake_minimum_required(VERSION 3.12)
project(KokkosComm VERSION 0.0.1)

option(KokkosComm_ENABLE_PERFTESTS "Build KokkosComm perf tests" ON)
set(KOKKOSCOMM_ENABLE_PERFTESTS ${KokkosComm_ENABLE_PERFTESTS} CACHE BOOL "" FORCE)
option(KokkosComm_ENABLE_TESTS "Build KokkosComm perf tests" ON)
set(KOKKOSCOMM_ENABLE_TESTS ${KokkosComm_ENABLE_TESTS} CACHE BOOL "" FORCE)

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


set(KOKKOSCOMM_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR} CACHE STRING "" FORCE)
set(KOKKOSCOMM_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR} CACHE STRING "" FORCE)
set(KOKKOSCOMM_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH} CACHE STRING "" FORCE)
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/KokkosComm_version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/src/KokkosComm_version.hpp
    @ONLY
)
target_include_directories(KokkosComm INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/src/)

enable_testing()
if(KOKKOSCOMM_ENABLE_TESTS)
    add_subdirectory(unit_tests)
endif()
if(KOKKOSCOMM_ENABLE_PERFTESTS)
    add_subdirectory(perf_tests)
endif()