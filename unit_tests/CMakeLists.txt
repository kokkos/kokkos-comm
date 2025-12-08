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
# If not included in the KokkosComm build, find KokkosComm to do a standalone build
if(NOT TARGET KokkosComm::KokkosComm)
  find_package(KokkosComm REQUIRED)
endif()
# MPI is used in standalone when setting up the GTest environment.
find_package(MPI REQUIRED)

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG
    52eb8108c5bdec04579160ae17225d66034bd723 # v1.17.0
)
# Don't install googletest
set(INSTALL_GTEST OFF)
FetchContent_MakeAvailable(googletest)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(gtest PUBLIC -w)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(gtest PUBLIC /w)
endif()

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG
    407c905e45ad75fc29bf0f9bb7c5c2fd3475976f #v12.1.0
)
FetchContent_MakeAvailable(fmt)

if(KOKKOSCOMM_ENABLE_MPI)
  # Standalone MPI smoke tests (do not use KokkosComm)
  add_executable(test-mpi)
  target_sources(test-mpi PRIVATE mpi/test_mpi.cpp)
  # doesn't use KokkosComm, so explicitly link MPI
  target_link_libraries(test-mpi PRIVATE MPI::MPI_CXX)
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
    target_link_libraries(test-mpi-cuda-sendrecv PRIVATE MPI::MPI_CXX)

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
    find_package(hip REQUIRED)
    add_executable(test-mpi-hip-sendrecv)
    target_sources(test-mpi-hip-sendrecv PRIVATE mpi/test_mpi_hip_sendrecv.cpp)
    target_link_libraries(test-mpi-hip-sendrecv PRIVATE MPI::MPI_CXX)
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

if(KOKKOSCOMM_ENABLE_NCCL)
  # Standalone NCCL smoke tests (do not use KokkosComm)
  add_executable(test-nccl)
  target_sources(test-nccl PRIVATE nccl/test_nccl.cpp)
  target_compile_features(test-nccl PRIVATE cxx_std_20)
  # doesn't use KokkosComm, so explicitly link with NCCL
  target_link_libraries(test-nccl PRIVATE NCCL::NCCL)
  add_test(NAME test-nccl-smoke COMMAND ./test-nccl)
endif()

# Tests using the MPI communication space
function(add_mpi_test test_name num_procs)
  # Extract source files from remaining arguments
  set(source_files ${ARGN})
  # Add test_main.cpp and the provided source files
  add_executable(${test_name})
  target_sources(
    ${test_name}
    PRIVATE
      test_main.cpp
      ${source_files}
  )
  target_link_libraries(
    ${test_name}
    PRIVATE
      KokkosComm::KokkosComm
      MPI::MPI_CXX # Explicitly link against MPI
      gtest
      fmt::fmt
  )
  add_test(
    NAME ${test_name}
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_procs} ./${test_name}
  )
endfunction()

# Tests using the NCCL communication space
function(add_nccl_test test_name num_procs)
  # Extract source files from remaining arguments
  set(source_files ${ARGN})
  # Add test_main.cpp and the provided source files
  add_executable(${test_name})
  target_sources(
    ${test_name}
    PRIVATE
      test_main.cpp
      ${source_files}
  )
  target_link_libraries(
    ${test_name}
    PRIVATE
      KokkosComm::KokkosComm
      NCCL::NCCL
      MPI::MPI_CXX # Explicitly link against MPI because it's needed to setup multi-node NCCL comms
      gtest
      fmt::fmt
  )
  add_test(
    NAME ${test_name}
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_procs} ./${test_name}
  )
endfunction()

# General KokkosComm tests
add_executable(test-main)
target_sources(
  test-main
  PRIVATE
    test_main.cpp
    test_sendrecv.cpp
  PRIVATE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES logging.hpp
)
target_compile_features(test-main PRIVATE cxx_std_20)
target_link_libraries(
  test-main
  PRIVATE
    GTest::gtest_main
    KokkosComm::KokkosComm
    MPI::MPI_CXX
    fmt::fmt
)

if(KokkosComm_ENABLE_MPI)
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

  add_test(
    NAME test-main
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ./test-main
  )
endif()

if(KokkosComm_ENABLE_NCCL)
  # Add `unit_tests/nccl/` to included directories
  target_sources(
    test-main
    PRIVATE
      FILE_SET HEADERS
      BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
      FILES nccl/utils.hpp
  )
  # Link against NCCL because we depend on raw calls
  target_link_libraries(test-main PRIVATE NCCL::NCCL)

  add_nccl_test(test-p2p       2 nccl/test_point_to_point.cpp)
  add_nccl_test(test-broadcast 2 nccl/test_broadcast.cpp)
  add_nccl_test(test-allgather 2 nccl/test_allgather.cpp)
  add_nccl_test(test-alltoall  2 nccl/test_alltoall.cpp)
  add_nccl_test(test-allreduce 2 nccl/test_allreduce.cpp)
  add_nccl_test(test-reduce    2 nccl/test_reduce.cpp)
endif()
