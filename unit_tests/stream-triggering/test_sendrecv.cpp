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
void send_comm_mode_1d_contig() {
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
  ViewType a("a", 1000); // send
  ViewType b("b", 1000); // recv

  StreamContext ctx;
  MPIS_Request my_request; 

  if (0 == rank) {
    int dst = 1;
    Kokkos::parallel_for(
    a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; }
			 );
    KokkosComm::Experimental::stream::send(a, 1, 0, MPI_COMM_WORLD, ctx.get_mem_info(), &my_request);
  }
  else if (1 == rank){
    KokkosComm::Experimental::stream::recv(b, 0, 0, MPI_COMM_WORLD, ctx.get_mem_info(), &my_request);
  }
  MPIS_Match(&my_request, MPI_STATUS_IGNORE);
  
  MPIS_Enqueue_startall( ctx.get_queue(), 1, &my_request );
  MPIS_Enqueue_waitall( ctx.get_queue() );

  if( 1 == rank) {
    int src = 0; int errs;
    Kokkos::parallel_reduce(
			    b.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += b(i) != i; }, errs);     
    ASSERT_EQ(errs, 0);
    std::cerr << "errs: " << errs << std::endl;
  }

  MPIS_Request_free(&my_request);

}

template <CommunicationMode SendMode, typename Scalar>
void send_comm_mode_1d_noncontig() {
  if constexpr (std::is_same_v<SendMode, CommModeReady>) {
    GTEST_SKIP() << "Skipping test for ready-mode send";
  }
  
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  
  using ViewType = typename Kokkos::View<Scalar **, Kokkos::LayoutRight>;
  // this is C-style layout, i.e. b(0,0) is next to b(0,1)
  ViewType base("base", 10, 10);
  ViewType base2("base2", 10, 10);
  auto a = Kokkos::subview(base, Kokkos::ALL, 2);  // take column 2 (non-contiguous)
  auto b = Kokkos::subview(base2, Kokkos::ALL, 2);  // take column 2 (non-contiguous)

  StreamContext ctx;
  MPIS_Request my_request; 
  if (0 == rank) {
    int dst = 1;
    Kokkos::parallel_for(
    a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; }
			 );
    KokkosComm::Experimental::stream::send(Kokkos::DefaultExecutionSpace(), a, 1, 0, MPI_COMM_WORLD, ctx, &my_request, DefaultCommMode{});
  }
  else if (1 == rank){
    KokkosComm::Experimental::stream::recv(Kokkos::DefaultExecutionSpace(), b, 0, 0, MPI_COMM_WORLD, ctx, &my_request);
  }
  //MPIS_Match(&my_request, MPI_STATUS_IGNORE);
  
  //MPIS_Enqueue_startall( ctx.get_queue(), 1, &my_request );
  //MPIS_Enqueue_waitall( ctx.get_queue() );

  if( 1 == rank) {
    int src = 0; int errs;
    Kokkos::parallel_reduce(
			    b.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += b(i) != i; }, errs);     
    ASSERT_EQ(errs, 0);
    std::cerr << "errs: " << errs << std::endl;
  }

  MPIS_Request_free(&my_request);

}
  
TYPED_TEST(MpiSendRecv, 1D_contig_standard) {
  send_comm_mode_1d_contig<CommModeStandard, typename TestFixture::Scalar>();
}

TYPED_TEST(MpiSendRecv, 1D_contig_ready) {
  send_comm_mode_1d_contig<CommModeReady, typename TestFixture::Scalar>();
}

  //  TYPED_TEST(MpiSendRecv, 1D_noncontig_standard) {
  //send_comm_mode_1d_noncontig<CommModeStandard, typename TestFixture::Scalar>();
  //}

}  // namespace
