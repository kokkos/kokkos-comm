cmake_minimum_required(VERSION 3.5)
project(KokkosComm)

find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

add_subdirectory(src)