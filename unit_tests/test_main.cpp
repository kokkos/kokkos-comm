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
#include <Kokkos_Core.hpp>
#include <KokkosComm.hpp>

class MpiEnvironment : public ::testing::Environment {
 public:
  ~MpiEnvironment() override {}

  // Override this to define how to set up the environment.
  void SetUp() override {}

  // Override this to define how to tear down the environment.
  void TearDown() override {}

  KokkosComm::Communicator comm_ = KokkosComm::CommWorld();
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
    auto comm = KokkosComm::CommWorld();
    int rank  = comm.rank();

    const int rankFailed = result.failed();
    if (rankFailed) {
      std::stringstream ss;
      ss << "(rank " << rank << " failed)";
      std::cout << ss.str() << std::endl;
    }

    // if one ranks has hung or crashed this reduce might not work, but most
    // of the info is hopefully printed above
    int globalFailed;
    comm.reduce(Kokkos::View<int const>{&rankFailed}, Kokkos::View<int>{&globalFailed}, KokkosComm::LogicalOr(), 0);
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

int main(int argc, char *argv[]) {
  // Intialize google test
  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);
  auto comm = KokkosComm::CommWorld();
  int rank  = comm.rank();
  if (0 == rank) {
    std::cerr << argv[0] << " (KokkosComm " << KOKKOSCOMM_VERSION_MAJOR << "." << KOKKOSCOMM_VERSION_MINOR << "."
              << KOKKOSCOMM_VERSION_PATCH << ")\n";
  }

  Kokkos::initialize();

  // Intialize google test
  ::testing::InitGoogleTest(&argc, argv);

  ::testing::AddGlobalTestEnvironment(new MpiEnvironment());

  auto &test_listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (0 != rank) delete test_listeners.Release(test_listeners.default_result_printer());

  test_listeners.Append(new MpiListener);

  // run tests
  auto exit_code = RUN_ALL_TESTS();

  // Finalize MPI before exiting
  Kokkos::finalize();
  MPI_Finalize();

  return exit_code;
}
