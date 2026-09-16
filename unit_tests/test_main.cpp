// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// https://google.github.io/googletest/advanced.html

#include <sstream>

#include <mpi.h>
#include <gtest/gtest.h>
#include <KokkosComm/config.hpp>
#include <Kokkos_Core.hpp>

#include "logging.hpp"
#ifdef KOKKOSCOMM_ENABLE_NCCL
#include "nccl/utils.hpp"
#endif

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
    printf("*** Test %s.%s ending.\n",
            test_info.test_suite_name(), test_info.name());
  }
#endif
};

int main(int argc, char* argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  KC_CHECK(provided == MPI_THREAD_MULTIPLE, "MPI_THREAD_MULTIPLE is required");
#ifdef KOKKOSCOMM_ENABLE_NCCL
  // Initialize the NCCL environment once for all tests (false = no verbose logs)
  test_utils::NcclCtx::init(false);
#endif
  Kokkos::initialize();
  ::testing::InitGoogleTest(&argc, argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (0 == rank) {
    KC_INFO(
        "{} - Kokkos Comm v{}.{}.{}-{} unit tests", argv[0], KOKKOSCOMM_VERSION_MAJOR, KOKKOSCOMM_VERSION_MINOR,
        KOKKOSCOMM_VERSION_PATCH, KOKKOSCOMM_COMMIT_HASH
    );
  }

  ::testing::AddGlobalTestEnvironment(new MpiEnvironment());
  auto& test_listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (0 != rank) delete test_listeners.Release(test_listeners.default_result_printer());
  test_listeners.Append(new MpiListener);

  // run tests
  auto exit_code = RUN_ALL_TESTS();

  Kokkos::finalize();
#ifdef KOKKOSCOMM_ENABLE_NCCL
  test_utils::NcclCtx::fini();
#endif
  MPI_Finalize();

  return exit_code;
}
