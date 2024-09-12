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

# if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#   # using Clang
# elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#   # using GCC
# elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
#   # using Intel C++
# elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
#   # using Visual Studio C++
# endif()

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag(-Wall CXX_HAS_WALL)
check_cxx_compiler_flag(-Wextra CXX_HAS_WEXTRA)
check_cxx_compiler_flag(-Wshadow CXX_HAS_WSHADOW)
check_cxx_compiler_flag(-Wpedantic CXX_HAS_WPEDANTIC)
check_cxx_compiler_flag(-pedantic CXX_HAS_PEDANTIC)
check_cxx_compiler_flag(-Wcast-align CXX_HAS_CAST_ALIGN)
check_cxx_compiler_flag(-Wformat=2 CXX_HAS_WFORMAT2)
check_cxx_compiler_flag(-Wmissing-include-dirs CXX_HAS_WMISSING_INCLUDE_DIRS)
check_cxx_compiler_flag(-Wno-gnu-zero-variadic-macro-arguments CXX_HAS_NO_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS)


function(kokkoscomm_add_cxx_flags)

    cmake_parse_arguments(ADD_CXX_FLAGS "INTERFACE" "TARGET" "" ${ARGN})

    if(ADD_CXX_FLAGS_INTERFACE)
        set(TARGET_COMPILE_OPTIONS_KEYWORD INTERFACE)
        set(TARGET_COMPILE_FEATURES_KEYWORD INTERFACE)
    else()
        set(TARGET_COMPILE_OPTIONS_KEYWORD PRIVATE)
        set(TARGET_COMPILE_FEATURES_KEYWORD PRIVATE)
    endif()

    if(CXX_HAS_WEXTRA)
        target_compile_options(${ADD_CXX_FLAGS_TARGET} ${TARGET_COMPILE_OPTIONS_KEYWORD} $<BUILD_INTERFACE:-Wextra>)
    endif()
    if(CXX_HAS_WSHADOW)
        target_compile_options(${ADD_CXX_FLAGS_TARGET} ${TARGET_COMPILE_OPTIONS_KEYWORD} $<BUILD_INTERFACE:-Wshadow>)
    endif()
    if(CXX_HAS_WPEDANTIC)
        target_compile_options(${ADD_CXX_FLAGS_TARGET} ${TARGET_COMPILE_OPTIONS_KEYWORD} $<BUILD_INTERFACE:-Wpedantic>)
    endif()
    if(NOT CXX_HAS_WPEDANTIC AND CXX_HAS_PEDANTIC)
        target_compile_options(${ADD_CXX_FLAGS_TARGET} ${TARGET_COMPILE_OPTIONS_KEYWORD} $<BUILD_INTERFACE:-pedantic>)
    endif()
    if(CXX_HAS_CAST_ALIGN)
        target_compile_options(${ADD_CXX_FLAGS_TARGET} ${TARGET_COMPILE_OPTIONS_KEYWORD} $<BUILD_INTERFACE:-Wcast-align>)
    endif()
    if(CXX_HAS_WFORMAT2)
        target_compile_options(${ADD_CXX_FLAGS_TARGET} ${TARGET_COMPILE_OPTIONS_KEYWORD} $<BUILD_INTERFACE:-Wformat=2>)
    endif()
    if(CXX_HAS_WMISSING_INCLUDE_DIRS)
        target_compile_options(${ADD_CXX_FLAGS_TARGET} ${TARGET_COMPILE_OPTIONS_KEYWORD} $<BUILD_INTERFACE:-Wmissing-include-dirs>)
    endif()
    # gtest includes sometimes yield this warning
    if(CXX_HAS_NO_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS)
        target_compile_options(${ADD_CXX_FLAGS_TARGET} ${TARGET_COMPILE_OPTIONS_KEYWORD} $<BUILD_INTERFACE:-Wno-gnu-zero-variadic-macro-arguments>)
    endif()

    # choose cxx standard
    set_target_properties(${ADD_CXX_FLAGS_TARGET} PROPERTIES CXX_EXTENSIONS OFF)
    target_compile_features(${ADD_CXX_FLAGS_TARGET} ${TARGET_COMPILE_FEATURES_KEYWORD} cxx_std_20)

endfunction()
