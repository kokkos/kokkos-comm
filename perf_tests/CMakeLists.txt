project(KokkosCommPerfTests VERSION 0.0.2)

# Treat the perf tests as a separate project
# If not included in the Kokkos Comm build,
# find Kokkos Comm to do a standalone build
if (NOT TARGET KokkosComm::KokkosComm)
  find_package(KokkosComm REQUIRED)
endif()

include(FetchContent)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
cmake_policy(SET CMP0135 NEW)
endif()

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  benchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
)
# FetchContent_MakeAvailable(benchmark)
# was making install install benchmark as well
# EXCLUDE_FROM_ALL here seems to be the magic
if (NOT benchmark_POPULATED)
  FetchContent_Populate(benchmark)
  add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
unset(BENCHMARK_ENABLE_TESTING)


add_executable(perf_test-main test_main.cpp
  test_sendrecv.cpp
  test_2dhalo.cpp
)
kokkoscomm_add_cxx_flags(TARGET perf_test-main)
target_link_libraries(perf_test-main KokkosComm::KokkosComm benchmark::benchmark)
add_test(NAME perf_test-main
         COMMAND mpirun -np 2 ./perf_test-main)


