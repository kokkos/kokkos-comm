// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <iostream>
#include <sstream>

#include <gtest/gtest.h>
#include <mpi.h>

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
    printf("*** Test %s.%s starting.\n", test_info.test_suite_name(), test_info.name());
  }
#endif

  // called after a failed assertion or SUCCESS()
  void OnTestPartResult(const testing::TestPartResult& result) override {
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
    printf("*** Test %s.%s ending.\n", test_info.test_suite_name(), test_info.name());
  }
#endif
};
