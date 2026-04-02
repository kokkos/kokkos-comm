cmake_minimum_required(VERSION 3.25)

project(
  KokkosCommUnitTests
  VERSION 0.2.0
  LANGUAGES CXX
  DESCRIPTION "Unit tests for the Kokkos Comm experimental explicit communication interfaces"
)

enable_testing()

# Treat the unit tests as a separate project
# If not included in the Kokkos Comm build, find Kokkos Comm to do a standalone build
if(NOT TARGET KokkosComm::KokkosComm)
  find_package(KokkosComm REQUIRED)
endif()
# MPI is used in standalone when setting up the GTest environment.
find_package(MPI REQUIRED)

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG
    52eb8108c5bdec04579160ae17225d66034bd723 # v1.17.0
)
# Don't install googletest
set(INSTALL_GTEST OFF)
FetchContent_MakeAvailable(googletest)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(gtest PUBLIC -w)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(gtest PUBLIC /w)
endif()

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG
    407c905e45ad75fc29bf0f9bb7c5c2fd3475976f #v12.1.0
)
FetchContent_MakeAvailable(fmt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(kc-test)

# --- Core API unit tests --- #
kc_add_unit_test(test.core.send-recv CORE NUM_PES 2 FILES test_main.cpp test_send_recv.cpp)
kc_add_unit_test(test.core.broadcast CORE NUM_PES 2 FILES test_main.cpp test_broadcast.cpp)
kc_add_unit_test(test.core.all-gather CORE NUM_PES 2 FILES test_main.cpp test_allgather.cpp)
kc_add_unit_test(test.core.reduce CORE NUM_PES 2 FILES test_main.cpp test_reduce.cpp)
kc_add_unit_test(test.core.all-reduce CORE NUM_PES 2 FILES test_main.cpp test_allreduce.cpp)
kc_add_unit_test(test.core.all-to-all CORE NUM_PES 2 FILES test_main.cpp test_alltoall.cpp)

kc_add_unit_test(test.core.communicator CORE NUM_PES 2 FILES test_main.cpp test_communicator.cpp)
kc_add_unit_test(test.core.datatype-conv CORE NUM_PES 1 FILES test_main.cpp test_datatype_conversion.cpp)
kc_add_unit_test(test.core.red-op-conv CORE NUM_PES 1 FILES test_main.cpp test_red_op_conversion.cpp)

# --- MPI backend unit tests --- #
if(KokkosComm_ENABLE_MPI)
  # Standalone MPI smoke tests (do not use KokkosComm)
  kc_add_unit_test(smoke.mpi.1pe MPI NUM_PES 1 FILES mpi/test_mpi.cpp)
  kc_add_unit_test(smoke.mpi.2pes MPI NUM_PES 2 FILES mpi/test_mpi.cpp)

  # KOKKOS_ENABLE_CUDA is not set?
  if(Kokkos_ENABLE_CUDA)
    kc_add_unit_test(smoke.mpi.cuda-p2p MPI NUM_PES 2 FILES mpi/test_mpi_cuda_sendrecv.cpp)
    # MPICH needs dynamic cudart (we think, pmodels/mpich#7304)
    if(KOKKOSCOMM_IMPL_MPI_IS_MPICH)
      # Setting the CUDA_RUNTIME_LIBRARY property on this target to "Shared" doesn't work for
      # reasons I don't understand.
      target_compile_options(smoke.mpi.cuda-p2p PUBLIC --cudart shared)
      target_link_options(smoke.mpi.cuda-p2p PUBLIC --cudart shared)
    endif()
  endif()

  if(Kokkos_ENABLE_HIP)
    find_package(hip REQUIRED)
    kc_add_unit_test(smoke.mpi.hip-p2p MPI NUM_PES 2 FILES mpi/test_mpi_hip_sendrecv.cpp OPTIONS -xhip LIBRARIES hip::host)
  endif()

  # Make sure GTest is working
  kc_add_unit_test(
    test.mpi.gtest
    MPI
    NUM_PES 2
    FILES test_main.cpp mpi/test_gtest_mpi.cpp
    LIBRARIES KokkosComm::KokkosComm
  )
  # Make sure MPI can access Kokkos::View data
  kc_add_unit_test(
    test.mpi.view-access
    MPI
    NUM_PES 2
    FILES test_main.cpp mpi/test_mpi_view_access.cpp
    LIBRARIES KokkosComm::KokkosComm
  )

  kc_add_unit_test(
    test.mpi.send-recv
    MPI
    NUM_PES 2
    FILES test_main.cpp mpi/test_sendrecv.cpp
    LIBRARIES KokkosComm::KokkosComm
  )
  kc_add_unit_test(
    test.mpi.isend-recv
    MPI
    NUM_PES 2
    FILES test_main.cpp mpi/test_isendrecv.cpp
    LIBRARIES KokkosComm::KokkosComm
  )
  kc_add_unit_test(
    test.mpi.barrier
    MPI
    NUM_PES 2
    FILES test_main.cpp mpi/test_barrier.cpp
    LIBRARIES KokkosComm::KokkosComm
  )
  kc_add_unit_test(
    test.mpi.broadcast
    MPI
    NUM_PES 2
    FILES test_main.cpp mpi/test_broadcast.cpp
    LIBRARIES KokkosComm::KokkosComm
  )
  kc_add_unit_test(
    test.mpi.reduce
    MPI
    NUM_PES 2
    FILES test_main.cpp mpi/test_reduce.cpp
    LIBRARIES KokkosComm::KokkosComm
  )
  kc_add_unit_test(
    test.mpi.ireduce
    MPI
    NUM_PES 2
    FILES test_main.cpp mpi/test_ireduce.cpp
    LIBRARIES KokkosComm::KokkosComm
  )
  kc_add_unit_test(
    test.mpi.all-reduce
    MPI
    NUM_PES 2
    FILES test_main.cpp mpi/test_allreduce.cpp
    LIBRARIES KokkosComm::KokkosComm
  )
  kc_add_unit_test(
    test.mpi.all-to-all
    MPI
    NUM_PES 2
    FILES test_main.cpp mpi/test_alltoall.cpp
    LIBRARIES KokkosComm::KokkosComm
  )
  kc_add_unit_test(
    test.mpi.all-gather
    MPI
    NUM_PES 2
    FILES test_main.cpp mpi/test_allgather.cpp
    LIBRARIES KokkosComm::KokkosComm
  )
  kc_add_unit_test(test.mpi.scan MPI NUM_PES 2 FILES test_main.cpp mpi/test_scan.cpp LIBRARIES KokkosComm::KokkosComm)
  kc_add_unit_test(
    test.mpi.channel-p2p
    MPI
    NUM_PES 2
    FILES test_main.cpp mpi/test_channel.cpp
    LIBRARIES KokkosComm::KokkosComm
  )
endif()

# --- NCCL backend unit tests --- #
if(KokkosComm_ENABLE_NCCL)
  kc_add_unit_test(smoke.nccl.1pe-multi_gpu NCCL NUM_PES 1 FILES nccl/test_nccl.cpp)

  kc_add_unit_test(
    test.nccl.p2p
    NCCL
    NUM_PES 2
    FILES test_main.cpp nccl/test_point_to_point.cpp
    LIBRARIES KokkosComm::KokkosComm MPI::MPI_CXX
  )
  kc_add_unit_test(
    test.nccl.broadcast
    NCCL
    NUM_PES 2
    FILES test_main.cpp nccl/test_broadcast.cpp
    LIBRARIES KokkosComm::KokkosComm MPI::MPI_CXX
  )
  kc_add_unit_test(
    test.nccl.all-gather
    NCCL
    NUM_PES 2
    FILES test_main.cpp nccl/test_allgather.cpp
    LIBRARIES KokkosComm::KokkosComm MPI::MPI_CXX
  )
  kc_add_unit_test(
    test.nccl.all-to-all
    NCCL
    NUM_PES 2
    FILES test_main.cpp nccl/test_alltoall.cpp
    LIBRARIES KokkosComm::KokkosComm MPI::MPI_CXX
  )
  kc_add_unit_test(
    test.nccl.all-reduce
    NCCL
    NUM_PES 2
    FILES test_main.cpp nccl/test_allreduce.cpp
    LIBRARIES KokkosComm::KokkosComm MPI::MPI_CXX
  )
  kc_add_unit_test(
    test.nccl.reduce
    NCCL
    NUM_PES 2
    FILES test_main.cpp nccl/test_reduce.cpp
    LIBRARIES KokkosComm::KokkosComm MPI::MPI_CXX
  )
endif()
