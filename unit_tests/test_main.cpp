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

#include <mpi.h>

#include <gtest/gtest.h>
// #include <gtest_mpi/gtest_mpi.hpp>

#include <KokkosComm.hpp>
#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {
#if 0
  // Initialize MPI before any call to gtest_mpi
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (0 == rank) {
    std::cerr << argv[0] << " (KokkosComm " << KOKKOSCOMM_VERSION_MAJOR << "." << KOKKOSCOMM_VERSION_MINOR << "." << KOKKOSCOMM_VERSION_PATCH << ")\n";
      std::cerr << "size=" << size << "\n";
  }
  Kokkos::initialize();
  if (0 == rank) std::cerr << __FILE__ << ":" << __LINE__ << " did initialize()\n";

  // Intialize google test
  ::testing::InitGoogleTest(&argc, argv);
  if (0 == rank) std::cerr << __FILE__ << ":" << __LINE__ << " did ::testing::InitGoogleTest()\n";

  // Add a test environment, which will initialize a test communicator
  // (a duplicate of MPI_COMM_WORLD)
  ::testing::AddGlobalTestEnvironment(new gtest_mpi::MPITestEnvironment());
  if (0 == rank) std::cerr << __FILE__ << ":" << __LINE__ << " did testing::AddGlobalTestEnvironment()\n";

  auto& test_listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (0 == rank) std::cerr << __FILE__ << ":" << __LINE__ << " did ::testing::UnitTest::GetInstance()->listeners()\n";

  // Remove default listener and replace with the custom MPI listener
  delete test_listeners.Release(test_listeners.default_result_printer());
  if (0 == rank) std::cerr << __FILE__ << ":" << __LINE__ << " removed default listener\n";
  test_listeners.Append(new gtest_mpi::PrettyMPIUnitTestResultPrinter());
  if (0 == rank) std::cerr << __FILE__ << ":" << __LINE__ << " added MPI listener\n";

  // run tests
  auto exit_code = RUN_ALL_TESTS();
  if (0 == rank) std::cerr << __FILE__ << ":" << __LINE__ << " ran tests\n";

  // Finalize MPI before exiting
  Kokkos::finalize();
  MPI_Finalize();

  return exit_code;
#else
  // Intialize google test
  ::testing::InitGoogleTest(&argc, argv);

  // Initialize MPI before any call to gtest_mpi
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Kokkos::initialize();

  auto &test_listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (0 != rank)
    delete test_listeners.Release(test_listeners.default_result_printer());

  // run tests
  auto exit_code = RUN_ALL_TESTS();

  // Finalize MPI before exiting
  Kokkos::finalize();
  MPI_Finalize();

  return exit_code;
#endif
}