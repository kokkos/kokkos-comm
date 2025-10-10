cmake_minimum_required(VERSION 3.23) # same as KokkosComm

project(
  KokkosCommUnitTests
  VERSION 0.2.0
  LANGUAGES
    CXX
  DESCRIPTION "Unit tests for the KokkosComm experimental communication interfaces"
)

enable_testing()

# Treat the unit tests as a separate project
# If not included in the Kokkos Comm build, find Kokkos Comm to do a standalone build
if(NOT TARGET KokkosComm::KokkosComm)
  find_package(KokkosComm REQUIRED)
endif()

include(FetchContent)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(
  googletest
  URL
    https://github.com/google/googletest/archive/a7f443b80b105f940225332ed3c31f2790092f47.zip # 05-28-2024
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest) was making install install googletest as well
# EXCLUDE_FROM_ALL here seems to be the magic
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(gtest PRIVATE -w)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(gtest PRIVATE /w)
endif()

if(KOKKOSCOMM_ENABLE_MPI)
  # Standalone MPI smoke tests (do not use KokkosComm)
  add_executable(test-mpi)
  target_sources(test-mpi PRIVATE mpi/test_mpi.cpp)
  # doesn't use KokkosComm, so explicitly link MPI
  target_link_libraries(test-mpi MPI::MPI_CXX)
  add_test(
    NAME test-mpi-1
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ./test-mpi
  )
  add_test(
    NAME test-mpi-2
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ./test-mpi
  )

  # KOKKOS_ENABLE_CUDA is not set?
  if(Kokkos_ENABLE_CUDA)
    add_executable(test-mpi-cuda-sendrecv)
    target_sources(test-mpi-cuda-sendrecv PRIVATE mpi/test_mpi_cuda_sendrecv.cpp)
    target_link_libraries(test-mpi-cuda-sendrecv MPI::MPI_CXX)

    # MPICH needs dynamic cudart (we think, pmodels/mpich#7304)
    if(KOKKOSCOMM_IMPL_MPI_IS_MPICH)
      # Setting the CUDA_RUNTIME_LIBRARY property on this target to "Shared" doesn't work for
      # reasons I don't understand.
      target_compile_options(
        test-mpi-cuda-sendrecv
        PUBLIC
          --cudart
          shared
      )
      target_link_options(
        test-mpi-cuda-sendrecv
        PUBLIC
          --cudart
          shared
      )
    endif()
    add_test(
      NAME test-mpi-cuda-sendrecv
      COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ./test-mpi-cuda-sendrecv
    )
  endif()

  if(Kokkos_ENABLE_HIP)
    add_executable(test-mpi-hip-sendrecv)
    target_sources(test-mpi-hip-sendrecv PRIVATE mpi/test_mpi_hip_sendrecv.cpp)
    target_link_libraries(test-mpi-hip-sendrecv MPI::MPI_CXX)
    find_package(hip REQUIRED)
    target_link_libraries(test-mpi-hip-sendrecv hip::host)
    target_compile_options(
      test-mpi-hip-sendrecv
      PUBLIC
        -x
        hip
    )
    add_test(
      NAME test-mpi-hip-sendrecv
      COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ./test-mpi-hip-sendrecv
    )
  endif()
endif()

# Tests using the MPI communication space, but not linking with MPI itself
function(add_mpi_test test_name num_procs)
  # Extract source files from remaining arguments
  set(sources ${ARGN})

  # Add test_main.cpp and the provided source files
  add_executable(${test_name})
  target_sources(
    ${test_name}
    PRIVATE
      test_main.cpp
      ${sources}
  )

  target_link_libraries(
    ${test_name}
    PRIVATE
      KokkosComm::KokkosComm
      gtest
  )

  add_test(
    NAME ${test_name}
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_procs} ./${test_name}
  )
endfunction()

add_mpi_test(test-gtest-mpi       2 mpi/test_gtest_mpi.cpp) # make sure gtest is working
add_mpi_test(test-mpi-view-access 2 mpi/test_mpi_view_access.cpp) # make sure MPI can access Kokkos::View data
add_mpi_test(test-sendrecv        2 mpi/test_sendrecv.cpp)
add_mpi_test(test-isendrecv       2 mpi/test_isendrecv.cpp)
add_mpi_test(test-barrier         2 mpi/test_barrier.cpp)
add_mpi_test(test-broadcast       2 mpi/test_broadcast.cpp)
add_mpi_test(test-reduce          2 mpi/test_reduce.cpp)
add_mpi_test(test-allreduce       2 mpi/test_allreduce.cpp)
add_mpi_test(test-alltoall        2 mpi/test_alltoall.cpp)
add_mpi_test(test-allgather       2 mpi/test_allgather.cpp)
add_mpi_test(test-scan            2 mpi/test_scan.cpp)
add_mpi_test(test-channel         2 mpi/test_channel.cpp)
