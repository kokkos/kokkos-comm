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

# Support two different build modes: 
#   Standalone
#   As a Trilinos package
if(COMMAND TRIBITS_PACKAGE_DECL)
  set(KOKKOSCOMM_HAS_TRILINOS ON CACHE BOOL "")
else()
  SET(KOKKOSCOMM_HAS_TRILINOS OFF CACHE BOOL "")
endif()
message(STATUS "KOKKOSCOMM_HAS_TRILINOS: ${KOKKOSCOMM_HAS_TRILINOS}")

# 3.9: MPI::MPI_CXX
# 3.12: CMAKE_PROJECT_VERSION_MAJOR
# 3.15: PUBLIC_HEADER files for interface libraries
# 3.23: FILE_SETs for easy export of header-only libraries
cmake_minimum_required(VERSION 3.23)

set(KOKKOSCOMM_VERSION_MAJOR 0 CACHE STRING "" FORCE)
set(KOKKOSCOMM_VERSION_MINOR 2 CACHE STRING "" FORCE)
set(KOKKOSCOMM_VERSION_PATCH 0 CACHE STRING "" FORCE)
set(KOKKOSCOMM_VERSION_STRING "${KOKKOSCOMM_VERSION_MAJOR}.${KOKKOSCOMM_VERSION_MINOR}.${KOKKOSCOMM_VERSION_PATCH}" CACHE STRING "" FORCE)

SET(PACKAGE_NAME KokkosComm)
SET(${PACKAGE_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
SET(${PACKAGE_NAME}_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if (KOKKOSCOMM_HAS_TRILINOS)
  tribits_package(KokkosComm)

  SET(PACKAGE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  STRING(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UC)
  
else()
  project(
    KokkosComm
    LANGUAGES
      CXX
    VERSION "${KOKKOSCOMM_VERSION_STRING}"
    DESCRIPTION "Experimental MPI interfaces (and more!) for the Kokkos C++ Performance Portability Programming ecosystem"
    HOMEPAGE_URL "https://kokkos.org/kokkos-comm/"
  )
endif()


include(cmake/mpi-vendor.cmake)

option(KokkosComm_ENABLE_PERFTESTS "Build KokkosComm perf tests" OFF)
option(KokkosComm_ENABLE_TESTS "Build KokkosComm tests" OFF)
option(KokkosComm_ENABLE_MPI "Build KokkosComm with MPI transport" ON)

# Resolve options
set(KOKKOSCOMM_ENABLE_PERFTESTS ${KokkosComm_ENABLE_PERFTESTS} CACHE BOOL "" FORCE)
set(KOKKOSCOMM_ENABLE_TESTS ${KokkosComm_ENABLE_TESTS} CACHE BOOL "" FORCE)
set(KOKKOSCOMM_ENABLE_MPI ${KokkosComm_ENABLE_MPI} CACHE BOOL "" FORCE)


if (KOKKOSCOMM_HAS_TRILINOS)
  # Trilinos will "find" Kokkos and MPI
else()
  find_package(Kokkos REQUIRED)
  if(KOKKOSCOMM_ENABLE_MPI)
    find_package(MPI REQUIRED)
  endif()
endif()

kokkoscomm_set_mpi_vendor_variables()

add_subdirectory(src/KokkosComm)
if(KOKKOSCOMM_ENABLE_TESTS)
  enable_testing() # so we can run ctest at the top-level build dir
  add_subdirectory(unit_tests)
endif()
if(KOKKOSCOMM_ENABLE_PERFTESTS)
  enable_testing() # so we can run ctest at the top-level build dir
  add_subdirectory(perf_tests)
endif()

# -- PACKAGING -- #
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Generate version config header

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/KokkosComm_config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/src/KokkosComm/config.hpp
  @ONLY
)

# required by TriBITS
if (KOKKOSCOMM_HAS_TRILINOS)
  tribits_package_postprocess()
else()
  # Generate package config file
  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/KokkosComm/KokkosCommConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosComm
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )

  # Generate package version file
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/KokkosComm/KokkosCommVersion.cmake
    VERSION "${KOKKOSCOMM_VERSION_STRING}"
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
endif()