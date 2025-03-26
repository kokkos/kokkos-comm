
add_executable(01_mvp_example 01_mvp.cpp)
target_link_libraries(01_mvp_example PRIVATE KokkosComm::KokkosComm)