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

set(TARGET_LIBRARY kokkoscomm)

include(${PROJECT_SOURCE_DIR}/cmake/flags.cmake)

add_library(KokkosComm INTERFACE)
add_library(KokkosComm::KokkosComm ALIAS KokkosComm)

target_sources(
  KokkosComm
  INTERFACE
    FILE_SET ${TARGET_LIBRARY}_public_headers
    TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/KokkosComm.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/collective.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/concepts.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/fwd.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/point_to_point.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/traits.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/config.hpp # Generated at configuration-time
)
target_sources(
  KokkosComm
  PRIVATE
    FILE_SET ${TARGET_LIBRARY}_impl_headers
    TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES ${PROJECT_SOURCE_DIR}/src/KokkosComm/impl/KokkosComm_contiguous.hpp
)
target_sources(
  KokkosComm
  INTERFACE
    FILE_SET ${TARGET_LIBRARY}_mpi_headers
    TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/mpi.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/allgather.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/alltoall.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/barrier.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/commmode.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/handle.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/irecv.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/isend.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/recv.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/reduce.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/req.hpp
      ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/send.hpp
)
target_sources(
  KokkosComm
  PRIVATE
    FILE_SET ${TARGET_LIBRARY}_mpi_impl_headers
    TYPE HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/impl/include_mpi.hpp
    FILES ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/impl/pack_traits.hpp
    FILES ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/impl/packer.hpp
    FILES ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/impl/tags.hpp
    FILES ${PROJECT_SOURCE_DIR}/src/KokkosComm/mpi/impl/types.hpp
)

# Compile flags & features
kokkoscomm_add_cxx_flags(TARGET KokkosComm INTERFACE)

# Linking
target_link_libraries(KokkosComm INTERFACE Kokkos::kokkos)
if(DEFINED KOKKOSCOMM_ENABLE_MPI)
  target_link_libraries(KokkosComm INTERFACE MPI::MPI_CXX)
endif()

# -- PACKAGING -- #
include(GNUInstallDirs)

# Generate version header file
set(KOKKOSCOMM_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR} CACHE STRING "" FORCE)
set(KOKKOSCOMM_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR} CACHE STRING "" FORCE)
set(KOKKOSCOMM_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH} CACHE STRING "" FORCE)
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/KokkosComm_config.hpp.in
  ${PROJECT_SOURCE_DIR}/src/KokkosComm/config.hpp
  @ONLY
)

# Install all headers
install(
  TARGETS
    ${PROJECT_NAME}
  EXPORT KokkosCommTargets
  FILE_SET
  ${TARGET_LIBRARY}_public_headers
  FILE_SET
  ${TARGET_LIBRARY}_impl_headers
  FILE_SET
  ${TARGET_LIBRARY}_mpi_headers
  FILE_SET
  ${TARGET_LIBRARY}_mpi_impl_headers
)
