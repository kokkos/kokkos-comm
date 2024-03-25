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

#include <gtest/gtest.h>

#include <KokkosComm.hpp>
#include <Kokkos_Core.hpp>

#include "KokkosComm_include_mpi.hpp"

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
  void OnTestPartResult(const testing::TestPartResult &result) override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int globalFailed;
    int rankFailed = result.failed();
    MPI_Reduce(&rankFailed, &globalFailed, 1, MPI_INT, MPI_LOR, 0,
               MPI_COMM_WORLD);

    if (globalFailed && 0 == rank) {
      std::cout << "[rank " << rank << "]: Failed" << std::endl;
    }
  }
};

int main(int argc, char *argv[]) {
  // Intialize google test
  ::testing::InitGoogleTest(&argc, argv);

  // Initialize MPI before any call to gtest_mpi
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (0 == rank) {
    std::cerr << argv[0] << " (KokkosComm " << KOKKOSCOMM_VERSION_MAJOR << "."
              << KOKKOSCOMM_VERSION_MINOR << "." << KOKKOSCOMM_VERSION_PATCH
              << ")\n";
    std::cerr << "size=" << size << "\n";
  }

  Kokkos::initialize();

  // Intialize google test
  ::testing::InitGoogleTest(&argc, argv);

  ::testing::AddGlobalTestEnvironment(new MpiEnvironment());

  auto &test_listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (0 != rank)
    delete test_listeners.Release(test_listeners.default_result_printer());

  test_listeners.Append(new MpiListener);

  // run tests
  auto exit_code = RUN_ALL_TESTS();

  // Finalize MPI before exiting
  Kokkos::finalize();
  MPI_Finalize();

  return exit_code;
}