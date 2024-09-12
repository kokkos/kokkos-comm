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

add_executable(perf-test-main)
target_sources(
  perf-test-main
  PRIVATE
    test_main.cpp
    test_sendrecv.cpp
    test_2dhalo.cpp
    test_osu_latency.cpp
)
if(KOKKOSCOMM_ENABLE_TESTS)
  kokkoscomm_add_cxx_flags(TARGET perf-test-main)
endif()
target_link_libraries(
  perf-test-main
  KokkosComm::KokkosComm
  benchmark::benchmark
)
add_test(
  NAME perf-test-main
  COMMAND
    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ./perf-test-main
)
