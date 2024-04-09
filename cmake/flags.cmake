include(CheckCXXCompilerFlag)

macro(check_and_add_flag)
    set(target ${ARGV0})
    set(flag ${ARGV1})

    check_cxx_compiler_flag(${flag} HAS_${flag})
    if (HAS_${flag})
        target_compile_options(${target} INTERFACE $<BUILD_INTERFACE:${flag}>)
    endif()
endmacro()

add_library(KokkosCommFlags INTERFACE)

check_and_add_flag(KokkosCommFlags -Wall)
check_and_add_flag(KokkosCommFlags -Wextra)
check_and_add_flag(KokkosCommFlags -Wshadow)
check_and_add_flag(KokkosCommFlags -Wpedantic)
check_and_add_flag(KokkosCommFlags -pedantic)
check_and_add_flag(KokkosCommFlags -Wcast-align)
check_and_add_flag(KokkosCommFlags -Wformat=2)
check_and_add_flag(KokkosCommFlags -Wmissing-include-dirs)
check_and_add_flag(KokkosCommFlags -Wno-gnu-zero-variadic-macro-arguments)

# choose cxx standard
set_target_properties(KokkosCommFlags PROPERTIES CXX_EXTENSIONS OFF)
target_compile_features(KokkosCommFlags INTERFACE cxx_std_20)

add_library(KokkosComm::KokkosCommFlags ALIAS KokkosCommFlags)
