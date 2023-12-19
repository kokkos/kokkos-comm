include(FetchContent)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/530d5c8c84abd2a46f38583ee817743c9b3a42b4.zip # 12-18-2023
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest)
# was making install install googletest as well
# EXCLUDE_FROM_ALL here seems to be the magic
if (NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# FetchContent_Declare(
#   mpitest
#   URL https://github.com/AdhocMan/gtest_mpi/archive/refs/tags/v1.1.0.zip
# )
# FetchContent_MakeAvailable(mpitest)

add_executable(test-mpi test_mpi.cpp)
kokkoscomm_add_cxx_flags(TARGET test-mpi)
add_test(NAME test-mpi-1
        COMMAND mpirun -np 1 ./test-mpi
)
add_test(NAME test-mpi-2
        COMMAND mpirun -np 2 ./test-mpi
)

add_executable(test-main test_main.cpp
  test_isendrecv.cpp
  test_reduce.cpp
  test_sendrecv.cpp
)
target_link_libraries(test-main KokkosComm gtest 
#gtest_mpi
)
kokkoscomm_add_cxx_flags(TARGET test-main)

add_test(NAME test-main
         COMMAND mpirun -np 2 ./test-main
)


