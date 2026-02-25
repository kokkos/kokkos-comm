cmake_minimum_required(VERSION 3.25)

project(
  KokkosComm
  LANGUAGES CXX
  VERSION 0.2.0
  DESCRIPTION
    "Experimental explicit communication interfaces for the Kokkos C++ Performance Portability Programming ecosystem"
  HOMEPAGE_URL "https://kokkos.org/kokkos-comm/"
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- OPTIONS --- #
option(KokkosComm_ENABLE_PERFTESTS "Build Kokkos Comm perf tests" OFF)
option(KokkosComm_ENABLE_TESTS "Build Kokkos Comm unit tests" OFF)
option(KokkosComm_INSTALL_CMAKE_PACKAGE "Create and install a CMake config-file package" ${PROJECT_IS_TOP_LEVEL})

option(KokkosComm_ENABLE_MPI "Build Kokkos Comm with MPI backend" ON)
option(KokkosComm_ENABLE_NCCL "Build Kokkos Comm with NCCL backend" OFF)
option(KokkosComm_ABORT_ON_ERROR "Runtime error checks trigger a global abort" OFF)

# Resolve options that must be PP-defined
set(KOKKOSCOMM_ENABLE_MPI ${KokkosComm_ENABLE_MPI})
set(KOKKOSCOMM_ENABLE_NCCL ${KokkosComm_ENABLE_NCCL})
set(KOKKOSCOMM_ABORT_ON_ERROR ${KokkosComm_ABORT_ON_ERROR})

message(STATUS "Kokkos Comm version: ${CMAKE_PROJECT_VERSION}")
set(KOKKOSCOMM_ENABLED_BACKENDS "")

# --- DEPENDENCIES --- #
# Require Kokkos 4.7+
find_package(Kokkos 4.7.0 REQUIRED)
if(KokkosComm_ENABLE_MPI)
  find_package(MPI REQUIRED)
  include(mpi-vendor)
  kokkoscomm_set_mpi_vendor_variables()
  list(APPEND KOKKOSCOMM_ENABLED_BACKENDS "MPI")
endif()
if(KokkosComm_ENABLE_NCCL)
  # Require NCCL 2+
  find_package(NCCL 2.0 REQUIRED)
  list(APPEND KOKKOSCOMM_ENABLED_BACKENDS "NCCL")
endif()

if(NOT KOKKOSCOMM_ENABLED_BACKENDS)
  message(FATAL_ERROR "No Kokkos Comm backends enabled! Enable at least one of: MPI, NCCL")
else()
  list(JOIN KOKKOSCOMM_ENABLED_BACKENDS ";" BACKENDS_STRING)
  message(STATUS "Enabled Kokkos Comm backends: ${BACKENDS_STRING}")
endif()

# --- LIBRARY --- #
add_library(KokkosComm INTERFACE)
add_subdirectory(src/KokkosComm)
add_library(KokkosComm::KokkosComm ALIAS KokkosComm)

# --- TESTING --- #
if(KokkosComm_ENABLE_TESTS)
  enable_testing() # so we can run ctest at the top-level build dir
  add_subdirectory(unit_tests)
endif()
if(KokkosComm_ENABLE_PERFTESTS)
  enable_testing() # so we can run ctest at the top-level build dir
  add_subdirectory(perf_tests)
endif()

# --- PACKAGING --- #
# Extract HEAD commit hash
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE KOKKOSCOMM_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Generate configuration header
configure_file(${PROJECT_SOURCE_DIR}/cmake/KokkosComm_config.hpp.in ${PROJECT_BINARY_DIR}/src/KokkosComm/config.hpp)

# Install the Kokkos Comm library (only headers)
install(TARGETS KokkosComm EXPORT KokkosCommTargets FILE_SET HEADERS)

# There are two possible scenarios:
#
# 1. Kokkos Comm is built and installed in a standalone, and the consuming project imports it via `find_package` in
#    its CMake build process.
#    In this case, the creation and installation of the CMake config-file package in the final build/install is
#    *enabled* by default. Installing the CMake config-file package ensures that all targets and variables
#    defined/configured during the standalone Kokkos Comm build are available to projects that `find_package` it.
#    See the related CMake documentation on config-file packages: https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html#libraries-providing-config-file-packages
#
# 2. Kokkos Comm is built via `add_subdirectory` or `FetchContent`, as part of the CMake build process of the consuming
#    project.
#    In this case, the creation and installation of the CMake config-file package in the final build/install is
#    *disabled* by default. Since the user builds Kokkos Comm alongside their project, it can be configured _in situ_.
#
# However, in both cases the headers are installed, since if the dependent project is itself installed, it will need to
# have the Kokkos Comm headers available (but not necessarily the CMake config-file package).
if(KokkosComm_INSTALL_CMAKE_PACKAGE)
  include(CMakePackageConfigHelpers)
  include(GNUInstallDirs)

  # Generate package config file
  configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in
    ${PROJECT_BINARY_DIR}/cmake/KokkosComm/KokkosCommConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosComm
    PATH_VARS PROJECT_NAME PROJECT_VERSION
  )

  # Generate package version file
  write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/cmake/KokkosComm/KokkosCommVersion.cmake
    COMPATIBILITY ExactVersion
    ARCH_INDEPENDENT # KokkosComm is a header-only library
  )

  # Install CMake config-file package
  install(
    FILES
      ${PROJECT_BINARY_DIR}/cmake/KokkosComm/KokkosCommConfig.cmake
      ${PROJECT_BINARY_DIR}/cmake/KokkosComm/KokkosCommVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosComm
  )
  install(
    EXPORT KokkosCommTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosComm
    FILE KokkosCommTargets.cmake
    NAMESPACE KokkosComm::
  )
endif()
