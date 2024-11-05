#@HEADER
# ************************************************************************
#
#                        Kokkos v. 4.0
#       Copyright (2022) National Technology & Engineering
#               Solutions of Sandia, LLC (NTESS).
#
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
#
# Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
# See https://kokkos.org/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#@HEADER

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
endif()

# Tests using the MPI communication space, but not linking with MPI itself
add_executable(test-main)
target_sources(
  test-main
  PRIVATE
    test_main.cpp
    mpi/test_gtest_mpi.cpp
    mpi/test_sendrecv.cpp
    mpi/test_allgather.cpp
    mpi/test_alltoall.cpp
    mpi/test_isendrecv.cpp
    mpi/test_alltoall.cpp
    mpi/test_reduce.cpp
    mpi/test_allgather.cpp
    mpi/test_waitall.cpp
    mpi/test_waitany.cpp
)
target_link_libraries(
  test-main
  PRIVATE
    KokkosComm::KokkosComm
    gtest
)
add_test(
  NAME test-main
  COMMAND
    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ./test-main
)
