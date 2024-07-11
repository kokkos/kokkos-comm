cmake_minimum_required(VERSION 3.12) # same as Kokkos Comm
project(KokkosCommUnitTests VERSION 0.0.2)
enable_testing()

# Treat the unit tests as a separate project
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

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/a7f443b80b105f940225332ed3c31f2790092f47.zip # 05-28-2024
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest) was making install install googletest as well
# EXCLUDE_FROM_ALL here seems to be the magic
FetchContent_GetProperties(googletest)
if (NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Standalone MPI smoke tests (do not use KokkosComm)


# Kokkos Comm tests
set(KOKKOSCOMM_TEST_SOURCES)
list(APPEND KOKKOSCOMM_TEST_SOURCES test_main.cpp
  test_sendrecv.cpp
  test_barrier.cpp

)
if(KOKKOSCOMM_ENABLE_MPI)

  add_executable(test-mpi mpi/test_mpi.cpp)
  add_test(NAME test-mpi-1
          COMMAND mpirun -np 1 ./test-mpi
  )
  add_test(NAME test-mpi-2
          COMMAND mpirun -np 2 ./test-mpi
  )
  # doesn't use KokkosComm, so explicitly link MPI
  target_link_libraries(test-mpi MPI::MPI_CXX)


  list(PREPEND KOKKOSCOMM_TEST_SOURCES
    mpi/test_gtest_mpi.cpp
  )
  list(APPEND KOKKOSCOMM_TEST_SOURCES
    mpi/test_sendrecv.cpp
    mpi/test_allgather.cpp
    mpi/test_alltoall.cpp
    mpi/test_isendrecv.cpp
    mpi/test_alltoall.cpp
    mpi/test_reduce.cpp
    mpi/test_allgather.cpp
  )
endif()
  add_executable(test-main ${KOKKOSCOMM_TEST_SOURCES}
)

target_link_libraries(test-main KokkosComm::KokkosComm gtest)
if(KOKKOSCOMM_ENABLE_TESTS)
  kokkoscomm_add_cxx_flags(TARGET test-main)
endif()

add_test(NAME test-main
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ./test-main
)


