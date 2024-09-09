add_executable(perf_test-main test_main.cpp
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