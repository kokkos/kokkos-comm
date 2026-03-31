add_executable(perf-test-main)
target_sources(
  perf-test-main
  PRIVATE
    test_main.cpp
    test_sendrecv.cpp
    test_channel_sendrecv.cpp
    test_2dhalo.cpp
    test_osu_latency.cpp
)
target_link_libraries(
  perf-test-main
  KokkosComm::KokkosComm
  benchmark::benchmark
  MPIAdvance::stream-triggering
)
#if(stream-triggering_FOUND)
 # target_link_libraries(KOKKOSCOMM INTERFACE stream-triggering)
#endif()

add_test(
  NAME perf-test-main
  COMMAND
    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ./perf-test-main
)

#option(KokkosComm_ENABLE_STREAM_COMM "Support from Stream-triggered Grid Halos" ON )
