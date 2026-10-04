// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <KokkosComm/KokkosComm.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <type_traits>
#include <utility>


// Moving a Request must transfer ownership of the underlying handle:
// after the moved-from object is destroyed, the moved-to object must still
// hold a valid (non-null) handle.
TEST(Request, MoveAssignKeepsHandleAfterSourceDestruction) {
  int send_value = 42;
  int recv_value = -1;

  MPI_Request recv_req = MPI_REQUEST_NULL;
  MPI_Request send_req = MPI_REQUEST_NULL;
  ASSERT_EQ(MPI_Irecv(&recv_value, 1, MPI_INT, 0, 123, MPI_COMM_SELF, &recv_req), MPI_SUCCESS);
  ASSERT_EQ(MPI_Isend(&send_value, 1, MPI_INT, 0, 123, MPI_COMM_SELF, &send_req), MPI_SUCCESS);

  KokkosComm::Request<KokkosComm::MpiSpace> persistent;
  {
    KokkosComm::Request<KokkosComm::MpiSpace> temporary(send_req);
    persistent = std::move(temporary);
  }  // temporary destroyed here

  // No wait() here: just inspect the handle, so a bug shows up as a failed
  // assertion instead of a segfault.
  EXPECT_FALSE(persistent.request() == MPI_REQUEST_NULL);

  // Clean up the raw MPI requests so nothing is pending at MPI_Finalize.
  ASSERT_EQ(MPI_Wait(&send_req, MPI_STATUS_IGNORE), MPI_SUCCESS);
  ASSERT_EQ(MPI_Wait(&recv_req, MPI_STATUS_IGNORE), MPI_SUCCESS);
  EXPECT_EQ(recv_value, 42);
}