// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <type_traits>

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <stream-triggering.h>

#include <KokkosComm/KokkosComm.hpp>

namespace {
  
using namespace KokkosComm::Experimental::stream;

template <typename T>
class MpiSendRecv : public testing::Test {
 public:
  using Scalar = T;
};
  
  using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(MpiSendRecv, ScalarTypes);

template <CommunicationMode SendMode, typename Scalar>
void pingpong_comm_mode_1d_contig() {
  if constexpr (std::is_same_v<SendMode, CommModeReady>) {
    GTEST_SKIP() << "Skipping test for ready-mode send";
  }
  
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  
  using ViewType = typename Kokkos::View<Scalar *>;
  ViewType send_buffer("send", 1000); // send
  ViewType recv_buffer("recv", 1000); // recv

  StreamContext ctx;
  // set up send and recv
  MPIS_Request my_reqs[2]; 
  if (0 == rank) {
    int dst = 1;
    Kokkos::parallel_for(
   send_buffer.extent(0), KOKKOS_LAMBDA(const int i) {send_buffer(i) = i; }
			 );
    KokkosComm::Experimental::stream::send(send_buffer, 1, 0, MPI_COMM_WORLD, ctx.get_mem_info(), &my_reqs[0]);
    KokkosComm::Experimental::stream::recv(recv_buffer, 1, 0, MPI_COMM_WORLD, ctx.get_mem_info(), &my_reqs[1]);
  }
  else if (1 == rank){
    KokkosComm::Experimental::stream::recv(recv_buffer, 0, 0, MPI_COMM_WORLD, ctx.get_mem_info(), &my_reqs[0]);
    KokkosComm::Experimental::stream::send(send_buffer, 0, 0, MPI_COMM_WORLD, ctx.get_mem_info(), &my_reqs[1]);
  }
  MPIS_Matchall(2, my_reqs, MPI_STATUS_IGNORE);
  
  MPIS_Enqueue_startall( ctx.get_queue(), 2, my_reqs );
  MPIS_Enqueue_waitall(ctx.get_queue());
  std::cerr << " after start all" << std::endl;

  if( 1 == rank) {
    int src = 0; int errs;
    Kokkos::parallel_reduce(
			    recv_buffer.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += recv_buffer(i) != i; }, errs);     
    ASSERT_EQ(errs, 0);
    std::cerr << "errs: " << errs << std::endl;
  }
}
  
TYPED_TEST(MpiSendRecv, 1D_contig_standard) {
  pingpong_comm_mode_1d_contig<CommModeStandard, typename TestFixture::Scalar>();
}

}  // namespace
