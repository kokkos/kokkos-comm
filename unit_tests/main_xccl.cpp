// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// https://google.github.io/googletest/advanced.html

#include <mpi.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "logging.hpp"
#include "test_env.hpp"
#include "nccl/utils.hpp"

int main(int argc, char* argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  KC_CHECK(provided == MPI_THREAD_MULTIPLE, "MPI_THREAD_MULTIPLE is required");

  // Initialize the xCCL environment once for all tests (false = no verbose logs)
  test_utils::XcclCtx::init(false);
  Kokkos::InitializationSettings kokkos_settings;
  kokkos_settings.set_device_id(test_utils::XcclCtx::get().device());
  Kokkos::initialize(kokkos_settings);
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
  test_utils::XcclCtx::fini();
  MPI_Finalize();

  return exit_code;
}
