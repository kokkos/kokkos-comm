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
FetchContent_MakeAvailable(benchmark)
unset(BENCHMARK_ENABLE_TESTING)


add_executable(perf_test-main test_main.cpp
  test_sendrecv.cpp
  test_2dhalo.cpp
)
target_link_libraries(perf_test-main KokkosComm benchmark::benchmark)
add_test(NAME perf_test-main
         COMMAND mpirun -np 2 ./perf_test-main)

