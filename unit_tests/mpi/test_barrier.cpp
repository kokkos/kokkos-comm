// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <KokkosComm/KokkosComm.hpp>

namespace {

TEST(Barrier, 0) { KokkosComm::mpi::barrier(MPI_COMM_WORLD); }

}  // namespace
