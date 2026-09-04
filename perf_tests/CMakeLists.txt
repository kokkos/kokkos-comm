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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(kc-test)

if(KokkosComm_ENABLE_MPI)
  kc_add_perf_test(bench.mpi.p2p MPI NUM_PES 2 FILES bench_main.cpp mpi/test_sendrecv.cpp LIBRARIES KokkosComm::KokkosComm)
  kc_add_perf_test(
    bench.mpi.channel-p2p
    MPI
    NUM_PES 2
    FILES bench_main.cpp mpi/bench_channel_p2p.cpp
    LIBRARIES KokkosComm::KokkosComm
  )
  kc_add_perf_test(bench.mpi.2d_halo MPI NUM_PES 2 FILES bench_main.cpp mpi/test_2d_halo.cpp LIBRARIES KokkosComm::KokkosComm)
  kc_add_perf_test(bench.mpi.osu_latency MPI NUM_PES 2 FILES bench_main.cpp mpi/test_osu_latency.cpp LIBRARIES KokkosComm::KokkosComm)
endif()
