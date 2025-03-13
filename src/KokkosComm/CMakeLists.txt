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

add_library(KokkosComm INTERFACE)
add_library(KokkosComm::KokkosComm ALIAS KokkosComm)

# Public headers
target_sources(
  KokkosComm
  INTERFACE
    FILE_SET kokkoscomm_public_headers
    TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES KokkosComm.hpp collective.hpp concepts.hpp fwd.hpp point_to_point.hpp traits.hpp
)

# Implementation detail headers
target_sources(
  KokkosComm
  INTERFACE FILE_SET kokkoscomm_impl_headers TYPE HEADERS BASE_DIRS ${PROJECT_SOURCE_DIR}/src FILES impl/contiguous.hpp
)

# Configuration header
target_sources(
  KokkosComm
  INTERFACE
    FILE_SET kokkoscomm_config_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_BINARY_DIR}/src
    FILES ${PROJECT_BINARY_DIR}/src/KokkosComm/config.hpp
)

if(KOKKOSCOMM_ENABLE_MPI)
  # Public MPI headers
  target_sources(
    KokkosComm
    INTERFACE
      FILE_SET kokkoscomm_mpi_headers
      TYPE HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES
        mpi/mpi_space.hpp
        mpi/comm_mode.hpp
        mpi/handle.hpp
        mpi/req.hpp
        mpi/irecv.hpp
        mpi/isend.hpp
        mpi/recv.hpp
        mpi/send.hpp
        mpi/allgather.hpp
        mpi/alltoall.hpp
        mpi/reduce.hpp
        mpi/barrier.hpp
  )

  # Implementation detail MPI headers
  target_sources(
    KokkosComm
    INTERFACE
      FILE_SET kokkoscomm_mpi_impl_headers
      TYPE HEADERS
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES mpi/impl/pack_traits.hpp mpi/impl/packer.hpp mpi/impl/tags.hpp mpi/impl/types.hpp
  )
endif()

# --- COMPILE FLAGS --- #
include(CheckCXXCompilerFlag)

macro(kokkoscomm_check_and_add_compile_options)
  set(target ${ARGV0})
  set(flag ${ARGV1})

  check_cxx_compiler_flag(${flag} HAS_${flag})
  if(HAS_${flag})
    target_compile_options(${target} INTERFACE ${flag})
  endif()
endmacro()

# KokkosComm is a C++20 project
add_library(KokkosCommFlags INTERFACE)
add_library(KokkosComm::KokkosCommFlags ALIAS KokkosCommFlags)
target_compile_features(KokkosCommFlags INTERFACE cxx_std_20)
set_target_properties(KokkosCommFlags PROPERTIES CXX_EXTENSIONS OFF)

kokkoscomm_check_and_add_compile_options(KokkosCommFlags -Wall)
kokkoscomm_check_and_add_compile_options(KokkosCommFlags -Wextra)
kokkoscomm_check_and_add_compile_options(KokkosCommFlags -Wshadow)
kokkoscomm_check_and_add_compile_options(KokkosCommFlags -Wpedantic)
kokkoscomm_check_and_add_compile_options(KokkosCommFlags -pedantic)
kokkoscomm_check_and_add_compile_options(KokkosCommFlags -Wcast-align)
kokkoscomm_check_and_add_compile_options(KokkosCommFlags -Wformat=2)
kokkoscomm_check_and_add_compile_options(KokkosCommFlags -Wmissing-src-dirs)
kokkoscomm_check_and_add_compile_options(KokkosCommFlags -Wno-gnu-zero-variadic-macro-arguments)

# Linking
# gh/pmodels/mpich#7304: MPICH 4.2.3 and 4.3.0 (and maybe earlier) rely on being
# linked before CUDA and friends, so link MPI before Kokkos.
# TODO_CMAKE: in CMake 3.31, we can use LINK_LIBRARIES_STRATEGY to explicitly ask CMake not to
# reorder how libraries are linked. In the mean time, we have to just hope (?).
if(KOKKOSCOMM_ENABLE_MPI)
  target_link_libraries(KokkosComm INTERFACE MPI::MPI_CXX)
endif()
target_link_libraries(KokkosComm INTERFACE KokkosComm::KokkosCommFlags Kokkos::kokkos)

# Install library
install(
  TARGETS KokkosComm KokkosCommFlags
  EXPORT KokkosCommTargets
  FILE_SET kokkoscomm_public_headers
  FILE_SET kokkoscomm_impl_headers
  FILE_SET kokkoscomm_mpi_headers
  FILE_SET kokkoscomm_mpi_impl_headers
  FILE_SET kokkoscomm_config_headers
)
