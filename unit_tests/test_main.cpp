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
  MPI_Comm comm_;
  int rank_;

 public:
  MpiListener(MPI_Comm comm) : comm_(comm) { MPI_Comm_rank(comm_, &rank_); }
  MpiListener() = delete;

  // Called before a test starts.
  void OnTestStart(const testing::TestInfo & /*test_info*/) override {
    // std::stringstream ss;
    // ss << "[" << rank_ << "] " << __FILE__ << ":" << __LINE__ << " " << test_info.name() << " start\n";
    // std::cerr << ss.str();
    MPI_Barrier(comm_);
  }

  void OnTestProgramStart(const testing::UnitTest &) override { MPI_Barrier(comm_); }
  void OnTestProgramEnd(const testing::UnitTest &) override { MPI_Barrier(comm_); }

#if 0
  void OnTestIterationStart(const testing::UnitTest &, int) override {
    // std::stringstream ss;
    // ss << "[" << rank_ << "] " << __FILE__ << ":" << __LINE__ << " " << "start\n";
    // std::cerr << ss.str();
    MPI_Barrier(comm_);
  }

  void OnTestIterationEnd(const testing::UnitTest &, int) override {
    // std::stringstream ss;
    // ss << "[" << rank_ << "] " << __FILE__ << ":" << __LINE__ << " " << "start\n";
    // std::cerr << ss.str();
    MPI_Barrier(comm_);
  }
#endif

  // called after a failed assertion or SUCCESS()
#if 0
  void OnTestPartResult(const testing::TestPartResult &result) override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int rankFailed = result.failed();
    if (rankFailed) {
      std::stringstream ss;
      ss << "(rank " << rank << " failed) ";
      ss << result.file_name() << ":" << result.line_number() << "\n";
      ss << result.message();
      std::cout << ss.str() << std::endl;
    }

    // if one ranks has hung or crashed this MPI_Reduce might not work, but most
    // of the info is hopefully printed above
    int globalFailed;
    MPI_Reduce(&rankFailed, &globalFailed, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
    if (globalFailed && (0 == rank)) {
      std::cout << "(some rank failed, more information above)" << std::endl;
    }

  }
#endif
  // Called after a test ends.
  void OnTestEnd(const testing::TestInfo & /*test_info*/) override {
    // std::cerr << __FILE__ << ":" << __LINE__ << " " << test_info.name() << " end\n";
    MPI_Barrier(comm_);
  }
};

int main(int argc, char *argv[]) {
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
              << KOKKOSCOMM_VERSION_PATCH << ") with " << size << " ranks\n";
  }

  Kokkos::initialize();

  // Initialize google test
  ::testing::InitGoogleTest(&argc, argv);

  ::testing::AddGlobalTestEnvironment(new MpiEnvironment());

  auto &test_listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (0 != rank) delete test_listeners.Release(test_listeners.default_result_printer());

  test_listeners.Append(new MpiListener{MPI_COMM_WORLD});

  // run tests
  auto exit_code = RUN_ALL_TESTS();

  // Finalize MPI before exiting
  Kokkos::finalize();
  MPI_Finalize();

  return exit_code;
}
