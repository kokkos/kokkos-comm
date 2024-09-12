add_executable(perf_test-main test_main.cpp
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

    test_sendrecv.cpp
    test_2dhalo.cpp
    test_osu_latency.cpp
  )
  if(KOKKOSCOMM_ENABLE_TESTS)
    kokkoscomm_add_cxx_flags(TARGET perf_test-main)
  endif()
  target_link_libraries(perf_test-main KokkosComm::KokkosComm benchmark::benchmark)
  add_test(NAME perf_test-main
          COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ./perf_test-main)