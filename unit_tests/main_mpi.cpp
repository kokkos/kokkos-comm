// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// https://google.github.io/googletest/advanced.html

#include <mpi.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/config.hpp>

#include <cstdlib>
#include <iostream>

#include "test_env.hpp"

int main(int argc, char* argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    std::cerr << "MPI_THREAD_MULTIPLE is required\n";
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    return EXIT_FAILURE;
  }
  Kokkos::initialize();
  ::testing::InitGoogleTest(&argc, argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (0 == rank) {
    std::cout << argv[0] << " - Kokkos Comm v" << KOKKOSCOMM_VERSION_MAJOR << "." << KOKKOSCOMM_VERSION_MINOR << "."
              << KOKKOSCOMM_VERSION_PATCH << "-" << KOKKOSCOMM_COMMIT_HASH << " unit tests\n";
  }

  ::testing::AddGlobalTestEnvironment(new MpiEnvironment());
  auto& test_listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (0 != rank) delete test_listeners.Release(test_listeners.default_result_printer());
  test_listeners.Append(new MpiListener);

  // run tests
  auto exit_code = RUN_ALL_TESTS();

  Kokkos::finalize();
  MPI_Finalize();

  return exit_code;
}
