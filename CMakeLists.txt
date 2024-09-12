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

# 3.9: MPI::MPI_CXX
# 3.12: CMAKE_PROJECT_VERSION_MAJOR
# 3.15: PUBLIC_HEADER files for interface libraries
# 3.23: FILE_SETs for easy export of header-only libraries
cmake_minimum_required(VERSION 3.23)

project(
  KokkosComm
  LANGUAGES
    CXX
  VERSION 0.2.0
  DESCRIPTION "Experimental MPI interfaces (and more!) for the Kokkos C++ Performance Portability Programming ecosystem"
)

option(KokkosComm_ENABLE_PERFTESTS "Build KokkosComm perf tests" OFF)
option(KokkosComm_ENABLE_TESTS "Build KokkosComm tests" OFF)
option(KokkosComm_ENABLE_MPI "Build KokkosComm with MPI transport" ON)

# Resolve options
set(KOKKOSCOMM_ENABLE_PERFTESTS ${KokkosComm_ENABLE_PERFTESTS} CACHE BOOL "" FORCE)
set(KOKKOSCOMM_ENABLE_TESTS ${KokkosComm_ENABLE_TESTS} CACHE BOOL "" FORCE)
set(KOKKOSCOMM_ENABLE_MPI ${KokkosComm_ENABLE_MPI} CACHE BOOL "" FORCE)

find_package(Kokkos REQUIRED)
message(STATUS "[KokkosComm] Kokkos_VERSION:              ${Kokkos_VERSION}")
if(DEFINED KOKKOSCOMM_ENABLE_MPI)
  find_package(MPI REQUIRED)
  message(STATUS "[KokkosComm] MPI_VERSION:                 ${MPI_VERSION}")
  message(STATUS "[KokkosComm] MPI_CXX_COMPILER:            ${MPI_CXX_COMPILER}")
  message(STATUS "[KokkosComm] MPI_CXX_COMPILE_OPTIONS:     ${MPI_CXX_COMPILE_OPTIONS}")
  message(STATUS "[KokkosComm] MPI_CXX_COMPILE_DEFINITIONS: ${MPI_CXX_COMPILE_DEFINITIONS}")
  message(STATUS "[KokkosComm] MPI_CXX_INCLUDE_DIRS:        ${MPI_CXX_INCLUDE_DIRS}")
  message(STATUS "[KokkosComm] MPI_CXX_LINK_FLAGS:          ${MPI_CXX_LINK_FLAGS}")
  message(STATUS "[KokkosComm] MPI_CXX_LIBRARIES:           ${MPI_CXX_LIBRARIES}")
endif()

add_subdirectory(src)
if(KOKKOSCOMM_ENABLE_TESTS)
  add_subdirectory(unit_tests)
endif()
if(KOKKOSCOMM_ENABLE_PERFTESTS)
  add_subdirectory(perf_tests)
endif()

# -- PACKAGING -- #
include(CMakePackageConfigHelpers)

# Generate package config file
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# Generate package version file
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}/${PROJECT_NAME}Version.cmake
  COMPATIBILITY SameMajorVersion
  ARCH_INDEPENDENT # KokkosComm is a header-only library
)

# Install CMake package config & version files
install(
  FILES
    ${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}/${PROJECT_NAME}Config.cmake
    ${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}/${PROJECT_NAME}Version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# Install KokkosComm library
install(
  EXPORT KokkosCommTargets
  FILE KokkosCommTargets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}
)
