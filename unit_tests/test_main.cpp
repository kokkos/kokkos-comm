//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

// https://google.github.io/googletest/advanced.html

#include <sstream>

#include <gtest/gtest.h>

#include <KokkosComm/config.hpp>
#include <Kokkos_Core.hpp>

#if defined(KOKKOSCOMM_ENABLE_MPI)
#include "KokkosComm/mpi/impl/include_mpi.hpp"

class MpiEnvironment : public ::testing::Environment {
 public:
  ~MpiEnvironment() override {}

  // Override this to define how to set up the environment.
  void SetUp() override { comm_ = MPI_COMM_WORLD; }

  // Override this to define how to tear down the environment.
  void TearDown() override {}

  MPI_Comm comm_;
};

class MpiListener : public testing::EmptyTestEventListener {
#if 0
  // Called before a test starts.
  void OnTestStart(const testing::TestInfo& test_info) override {
    printf("*** Test %s.%s starting.\n",
            test_info.test_suite_name(), test_info.name());
  }
#endif

  // called after a failed assertion or SUCCESS()
  void OnTestPartResult(const testing::TestPartResult &result) override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int rankFailed = result.failed();
    if (rankFailed) {
      std::stringstream ss;
      ss << "(rank " << rank << " failed)";
      std::cout << ss.str() << std::endl;
    }

    // if one ranks has hung or crashed this MPI_Reduce might not work, but most
    // of the info is hopefully printed above
    int globalFailed;
    MPI_Reduce(&rankFailed, &globalFailed, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
    if (globalFailed && 0 == rank) {
      std::cout << "(some rank failed, more information above)" << std::endl;
    }
  }

#if 0
  // Called after a test ends.
  void OnTestEnd(const testing::TestInfo& test_info) override {
    printf("*** Test %s.%s ending.\n",
            test_info.test_suite_name(), test_info.name());
  }
#endif
};
#endif  // KOKKOSCOMM_ENABLE_MPI

int main(int argc, char *argv[]) {
#if defined(KOKKOSCOMM_ENABLE_MPI)
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    throw std::runtime_error("MPI_THREAD_MULTIPLE is needed");
  }
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (0 == rank) {
    std::cerr << argv[0] << " (KokkosComm " << KOKKOSCOMM_VERSION_MAJOR << "." << KOKKOSCOMM_VERSION_MINOR << "."
              << KOKKOSCOMM_VERSION_PATCH << ")\n";
  }
#endif

  Kokkos::initialize();

  // Intialize google test
  ::testing::InitGoogleTest(&argc, argv);

#if defined(KOKKOSCOMM_ENABLE_MPI)
  ::testing::AddGlobalTestEnvironment(new MpiEnvironment());
  auto &test_listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (0 != rank) delete test_listeners.Release(test_listeners.default_result_printer());
  test_listeners.Append(new MpiListener);
#endif

  // run tests
  auto exit_code = RUN_ALL_TESTS();

  // Finalize MPI before exiting
  Kokkos::finalize();

#if defined(KOKKOSCOMM_ENABLE_MPI)
  MPI_Finalize();
#endif

  return exit_code;
}
