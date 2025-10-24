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

add_subdirectory(src/KokkosComm)

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
