cmake_minimum_required(VERSION 3.23) # same as KokkosComm

project(
  KokkosCommExamples
  VERSION 0.2.0
  LANGUAGES
    CXX
  DESCRIPTION "Examples for the KokkosComm experimental communication interfaces"
)

# Treat the unit tests as a separate project
# If not included in the Kokkos Comm build, find Kokkos Comm to do a standalone build
if(NOT TARGET KokkosComm::KokkosComm)
  find_package(KokkosComm REQUIRED)
endif()

add_subdirectory(01_mvp)
