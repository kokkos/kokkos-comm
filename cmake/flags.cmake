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
check_cxx_compiler_flag(-Wno-gnu-zero-variadic-macro-arguments CXX_HAS_NO_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS)

function(kokkoscomm_add_cxx_flags)

    message(STATUS ${ARGN})
    cmake_parse_arguments(ADD_CXX_FLAGS "INTERFACE" "TARGET" "" ${ARGN})

    if(ADD_CXX_FLAGS_INTERFACE)
        set(TARGET_COMPILE_OPTIONS_KEYWORD INTERFACE)
    else()
        set(TARGET_COMPILE_OPTIONS_KEYWORD PUBLIC)
    endif()

    if(CXX_HAS_WEXTRA)
        target_compile_options(${ADD_CXX_FLAGS_TARGET} ${TARGET_COMPILE_OPTIONS_KEYWORD} -Wextra)
    endif()
    if(CXX_HAS_WSHADOW)
        target_compile_options(${ADD_CXX_FLAGS_TARGET} ${TARGET_COMPILE_OPTIONS_KEYWORD} -Wshadow)
    endif()
    if(CXX_HAS_WPEDANTIC)
        target_compile_options(${ADD_CXX_FLAGS_TARGET} ${TARGET_COMPILE_OPTIONS_KEYWORD} -Wpedantic)
    endif()
    if(NOT CXX_HAS_WPEDANTIC AND CXX_HAS_PEDANTIC)
        target_compile_options(${ADD_CXX_FLAGS_TARGET} ${TARGET_COMPILE_OPTIONS_KEYWORD} -pedantic)
    endif()

    # gtest includes sometimes yield this warning
    if(CXX_HAS_NO_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS)
        target_compile_options(${ADD_CXX_FLAGS_TARGET} ${TARGET_COMPILE_OPTIONS_KEYWORD} -Wno-gnu-zero-variadic-macro-arguments)
    endif()

    # kokkos requires C++17
    set_property(TARGET ${ADD_CXX_FLAGS_TARGET} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${ADD_CXX_FLAGS_TARGET} PROPERTY CXX_EXTENSIONS OFF)

endfunction()