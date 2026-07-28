cmake_minimum_required(VERSION 3.25)

project(
  KokkosCommPerfTests
  VERSION 0.1.0
  LANGUAGES CXX
  DESCRIPTION "Performance tests for the Kokkos Comm experimental explicit communication interfaces"
)

enable_testing()

# Treat the perf tests as a separate project
# If not included in the Kokkos Comm build, find Kokkos Comm to do a standalone build
if(NOT TARGET KokkosComm::KokkosComm)
  find_package(KokkosComm REQUIRED)
endif()
find_package(MPI REQUIRED)
if(KokkosComm_ENABLE_NCCL)
  find_package(NCCL REQUIRED)
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

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG
    407c905e45ad75fc29bf0f9bb7c5c2fd3475976f #v12.1.0
)
FetchContent_MakeAvailable(fmt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(kc-test)

kc_add_perf_test(bench.core.latency CORE NUM_PES 2 FILES bench_main.cpp bench_latency.cpp)

if(KokkosComm_ENABLE_MPI)
  kc_add_perf_test(bench.mpi.p2p MPI NUM_PES 2 FILES bench_main.cpp mpi/test_sendrecv.cpp LIBRARIES KokkosComm::KokkosComm)
  kc_add_perf_test(bench.mpi.channel_p2p MPI NUM_PES 2 FILES bench_main.cpp mpi/test_channel_sendrecv.cpp LIBRARIES KokkosComm::KokkosComm)
  kc_add_perf_test(bench.mpi.2d_halo MPI NUM_PES 2 FILES bench_main.cpp mpi/test_2d_halo.cpp LIBRARIES KokkosComm::KokkosComm)
endif()
