cmake_minimum_required(VERSION 3.25)

project(
  KokkosCommPerfTests
  VERSION 0.2.0
  LANGUAGES CXX
  DESCRIPTION "Performance tests for the Kokkos Comm experimental explicit communication interfaces"
)

enable_testing()

# Treat the perf tests as a separate project
# If not included in the Kokkos Comm build, find Kokkos Comm to do a standalone build
if(NOT TARGET KokkosComm::KokkosComm)
  find_package(KokkosComm REQUIRED)
endif()

include(FetchContent)

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG
    192ef10025eb2c4cdd392bc502f0c852196baa48 # v1.9.5
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

if(KokkosComm_ENABLE_MPI)
  add_subdirectory(mpi)
endif()
