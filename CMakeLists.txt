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
  LANGUAGES CXX
  VERSION 0.2.0
  DESCRIPTION "Experimental MPI interfaces (and more!) for the Kokkos C++ Performance Portability Programming ecosystem"
  HOMEPAGE_URL "https://kokkos.org/kokkos-comm/"
)

option(KokkosComm_ENABLE_PERFTESTS "Build KokkosComm perf tests" OFF)
option(KokkosComm_ENABLE_TESTS "Build KokkosComm tests" OFF)
option(KokkosComm_ENABLE_MPI "Build KokkosComm with MPI transport" ON)
option(KokkosComm_ENABLE_NCCL "Build KokkosComm with NCCL transport" OFF)

# Resolve options
set(KOKKOSCOMM_ENABLE_PERFTESTS ${KokkosComm_ENABLE_PERFTESTS} CACHE BOOL "" FORCE)
set(KOKKOSCOMM_ENABLE_TESTS ${KokkosComm_ENABLE_TESTS} CACHE BOOL "" FORCE)
set(KOKKOSCOMM_ENABLE_MPI ${KokkosComm_ENABLE_MPI} CACHE BOOL "" FORCE)
set(KOKKOSCOMM_ENABLE_NCCL ${KokkosComm_ENABLE_NCCL} CACHE BOOL "" FORCE)

find_package(Kokkos REQUIRED)
if(KOKKOSCOMM_ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()
if(KOKKOSCOMM_ENABLE_NCCL)
  find_package(NCCL REQUIRED)
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
include(GNUInstallDirs)

# Generate version config header
set(KOKKOSCOMM_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR} CACHE STRING "" FORCE)
set(KOKKOSCOMM_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR} CACHE STRING "" FORCE)
set(KOKKOSCOMM_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH} CACHE STRING "" FORCE)
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/KokkosComm_config.hpp.in
  ${PROJECT_BINARY_DIR}/src/KokkosComm/config.hpp
  @ONLY
)

# Generate package config file
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/KokkosComm/KokkosCommConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosComm
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# Generate package version file
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/cmake/KokkosComm/KokkosCommVersion.cmake
  COMPATIBILITY SameMajorVersion
  ARCH_INDEPENDENT # KokkosComm is a header-only library
)

# Install CMake package files
install(
  FILES
    ${PROJECT_BINARY_DIR}/cmake/KokkosComm/KokkosCommConfig.cmake
    ${PROJECT_BINARY_DIR}/cmake/KokkosComm/KokkosCommVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosComm
)
install(
  EXPORT KokkosCommTargets
  FILE KokkosCommTargets.cmake
  NAMESPACE KokkosComm::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosComm
)
