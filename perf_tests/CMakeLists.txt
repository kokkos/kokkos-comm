cmake_minimum_required(VERSION 3.23) # same as KokkosComm

project(
  KokkosCommPerfTests
  VERSION 0.2.0
  LANGUAGES
    CXX
  DESCRIPTION "Performance tests for the KokkosComm experimental communication interfaces"
)

enable_testing()

# Treat the perf tests as a separate project
# If not included in the Kokkos Comm build, find Kokkos Comm to do a standalone build
if(NOT TARGET KokkosComm::KokkosComm)
  find_package(KokkosComm REQUIRED)
endif()

include(FetchContent)

FetchContent_Declare(benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG
    eddb0241389718a23a42db6af5f0164b6e0139af # v1.9.4
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

if(KOKKOSCOMM_ENABLE_MPI)
  add_subdirectory(mpi)
endif(KOKKOSCOMM_ENABLE_MPI)
# Find and enable stream triggering
option(KokkosComm_ENABLE_MPI_ADVANCE "Support MPI Advance backends." ON)
option(KokkosComm_ENABLE_STREAMCOMM "Support Stream-Triggered Communication." ON)
# find MPI ADVANCE Stream Triggering
if(KokkosComm_ENABLE_MPI_ADVANCE)
  find_package(stream-triggering REQUIRED)
endif()
set(KokkosComm_ENABLE_STREAMCOMM KokkosComm_ENABLE_STREAMCOMM_FOUND)
