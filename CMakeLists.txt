# 3.9: MPI::MPI_CXX
# 3.12: CMAKE_PROJECT_VERSION_MAJOR
# 3.15: PUBLIC_HEADER files for interface libraries
cmake_minimum_required(VERSION 3.12)
project(KokkosComm VERSION 0.0.2)

option(KokkosComm_ENABLE_PERFTESTS "Build KokkosComm perf tests" ON)
option(KokkosComm_ENABLE_TESTS "Build KokkosComm perf tests" ON)
option(KokkosComm_ENABLE_MDSPAN "Build mdspan overloads" OFF)
option(KokkosComm_MDSPAN_IMPL "mdspan implementation to use [std, kokkos]" "kokkos")
option(KokkosComm_USE_STD_MDSPAN "Use std::mdspan (requires c++23)" OFF)
option(KokkosComm_USE_KOKKOS_MDSPAN "Use Kokkos mdspan implementation" OFF)


## resolve options
set(KOKKOSCOMM_ENABLE_PERFTESTS ${KokkosComm_ENABLE_PERFTESTS} CACHE BOOL "" FORCE)
set(KOKKOSCOMM_ENABLE_TESTS ${KokkosComm_ENABLE_TESTS} CACHE BOOL "" FORCE)
set(KOKKOSCOMM_ENABLE_MDSPAN ${KokkosComm_ENABLE_MDSPAN} CACHE BOOL "" FORCE)
set(KOKKOSCOMM_USE_KOKKOS_MDSPAN ${KokkosComm_USE_KOKKOS_MDSPAN} CACHE BOOL "" FORCE)
set(KOKKOSCOMM_USE_STD_MDSPAN ${KokkosComm_USE_STD_MDSPAN} CACHE BOOL "" FORCE)

find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

message(STATUS "Kokkos MPI: MPI_VERSION =                 ${MPI_VERSION}")
message(STATUS "Kokkos MPI: MPI_CXX_COMPILER =            ${MPI_CXX_COMPILER}")
message(STATUS "Kokkos MPI: MPI_CXX_COMPILE_OPTIONS =     ${MPI_CXX_COMPILE_OPTIONS}")
message(STATUS "Kokkos MPI: MPI_CXX_COMPILE_DEFINITIONS = ${MPI_CXX_COMPILE_DEFINITIONS}")
message(STATUS "Kokkos MPI: MPI_CXX_INCLUDE_DIRS =        ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "Kokkos MPI: MPI_CXX_LINK_FLAGS =          ${MPI_CXX_LINK_FLAGS}")
message(STATUS "Kokkos MPI: MPI_CXX_LIBRARIES =           ${MPI_CXX_LIBRARIES}")
if (KOKKOSCOMM_ENABLE_MDSPAN)
    if (KOKKOSCOMM_USE_STD_MDSPAN)
        message(STATUS "Kokkos MPI: mdspan support enabled (std::mdspan)")
    elseif(KOKKOSCOMM_USE_KOKKOS_MDSPAN)
        message(STATUS "Kokkos MPI: mdspan support enabled (kokkos/mdspan)")
    else()
        message(FATAL_ERROR "Kokkos MPI: mdspan support enabled but no MDSPAN implementation enabled!")
    endif()
endif()

include(FetchContent)
if(KOKKOSCOMM_ENABLE_MDSPAN AND KOKKOSCOMM_USE_KOKKOS_MDSPAN)
    message(STATUS "retrieving kokkos/mdspan...")
    FetchContent_Declare(
        kokkos_mdspan
        GIT_REPOSITORY https://github.com/kokkos/mdspan.git
        GIT_TAG        9ceface91483775a6c74d06ebf717bbb2768452f # mdspan-0.6.0
        )
    FetchContent_MakeAvailable(kokkos_mdspan)
endif()


include(cmake/flags.cmake)
add_subdirectory(src)
kokkoscomm_add_cxx_flags(TARGET KokkosComm INTERFACE)
message(STATUS ${KOKKOSCOMM_PUBLIC_HEADERS})
set_target_properties(KokkosComm PROPERTIES PUBLIC_HEADER "${KOKKOSCOMM_PUBLIC_HEADERS}")

## Version config file
set(KOKKOSCOMM_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR} CACHE STRING "" FORCE)
set(KOKKOSCOMM_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR} CACHE STRING "" FORCE)
set(KOKKOSCOMM_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH} CACHE STRING "" FORCE)
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/KokkosComm_version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/src/KokkosComm_version.hpp
    @ONLY
)

# this sets CMAKE_INSTALL_[...]DIR
include(GNUInstallDirs)

target_include_directories(KokkosComm INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src/>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

## Package config file
include(CMakePackageConfigHelpers)
configure_package_config_file (
    ${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in
    ${PROJECT_BINARY_DIR}/cmake/KokkosComm/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/cmake/KokkosComm/${PROJECT_NAME}Version.cmake
    COMPATIBILITY SameMajorVersion
)

export (
    TARGETS KokkosComm
    NAMESPACE "${PROJECT_NAME}::"
    FILE ${PROJECT_BINARY_DIR}/cmake/KokkosComm/KokkosCommTargets.cmake
)


install(
    TARGETS KokkosComm
    EXPORT KokkosCommTargets # does this do anything?
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    DIRECTORY "${PROJECT_BINARY_DIR}/src/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")
install(
    FILES 
    "${PROJECT_BINARY_DIR}/cmake/KokkosComm/KokkosCommConfig.cmake"
    "${PROJECT_BINARY_DIR}/cmake/KokkosComm/KokkosCommVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosComm/
)
install(
    EXPORT KokkosCommTargets
    NAMESPACE "${PROJECT_NAME}::"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
)


if(KOKKOSCOMM_ENABLE_TESTS)
    add_subdirectory(unit_tests)
endif()
if(KOKKOSCOMM_ENABLE_PERFTESTS)
    add_subdirectory(perf_tests)
endif()