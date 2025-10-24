cmake_minimum_required(VERSION 3.25...4.2)

project(
  KokkosComm
  DESCRIPTION "Experimental communication interfaces for the Kokkos C++ Performance Portability Programming ecosystem"
  LANGUAGES
    CXX
  VERSION 0.2.0
  HOMEPAGE_URL "https://kokkos.org/kokkos-comm/"
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(mpi-vendor)

# --- OPTIONS --- #
option(KokkosComm_BUILD_UNITTESTS "Build unit tests. Default: OFF. Values: { ON, OFF }." OFF)
option(KokkosComm_BUILD_PERFTESTS "Build perf tests. Default: OFF. Values: { ON, OFF }." OFF)
option(
  KokkosComm_INSTALL_CMAKE_PACKAGE
  "Install a CMake config-file package. Default: ${PROJECT_IS_TOP_LEVEL}. Values: { ON, OFF }."
  ${PROJECT_IS_TOP_LEVEL}
)
option(KokkosComm_ENABLE_MPI "Enable MPI communication space backend. Default: ON. Values: { ON, OFF }." ON)
option(KokkosComm_ENABLE_NCCL "Enable NCCL communication space backend. Default: OFF. Values: { ON, OFF }." OFF)
option(
  KokkosComm_ENABLE_ABORT_ON_ERROR
  "Enable global abort on runtime error checks. Default: OFF. Values: { ON, OFF }."
  OFF
)

find_package(Kokkos REQUIRED)
if(KOKKOSCOMM_ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()
kokkoscomm_set_mpi_vendor_variables()
if(KOKKOSCOMM_ENABLE_NCCL)
  find_package(NCCL REQUIRED)
endif()

# --- LIBRARY --- #
add_library(KokkosComm INTERFACE)
add_subdirectory(src/KokkosComm)
add_library(KokkosComm::KokkosComm ALIAS KokkosComm)

# --- PACKAGING --- #
# Install library
install(TARGETS KokkosComm EXPORT KokkosCommTargets FILE_SET HEADERS)

if(KokkosComm_INSTALL_CMAKE_PACKAGE)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  # Generate package config-file
  configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/KokkosCommConfig.cmake.in
    ${PROJECT_BINARY_DIR}/KokkosCommConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosComm
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )

  # Generate package version-file
  write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/KokkosCommVersion.cmake
    COMPATIBILITY ExactVersion
    ARCH_INDEPENDENT # KokkosComm is a header-only library
  )

  # Install CMake package config files
  install(
    FILES
      ${PROJECT_BINARY_DIR}/KokkosCommConfig.cmake
      ${PROJECT_BINARY_DIR}/KokkosCommVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosComm
  )
  install(
    EXPORT KokkosCommTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosComm
    NAMESPACE KokkosComm::
  )
endif()

# --- TESTING / BENCHMARKING --- #
if(KokkosComm_BUILD_UNITTESTS)
  enable_testing()
  add_subdirectory(unit_tests)
endif()
if(KokkosComm_BUILD_PERFTESTS)
  # Allow registering benchmarks as tests so we can run them using `ctest`
  enable_testing()
  add_subdirectory(perf_tests)
endif()
