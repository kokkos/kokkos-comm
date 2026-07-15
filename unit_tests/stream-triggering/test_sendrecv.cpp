// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <type_traits>

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <stream-triggering.h>

#include <KokkosComm/KokkosComm.hpp>
//#include "utils.hpp"

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
  /*
  MPI_Info _mem_info;
  MPI_Info_create( &_mem_info );
  MPI_Info_set(_mem_info, "mpi_memory_alloc_kinds", "rocm:device:coarse");

  void* _my_stream;
  MPIS_Queue _my_queue;

  if constexpr ( std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::HIP>){
		 _my_stream = Kokkos::HIP().hip_stream();
  }
  else{
    _my_stream = nullptr;
  }
  MPIS_Queue_init(&_my_queue, CXI, &_my_stream);     
  */
  StreamContext ctx;
  // set up send and recv
  MPIS_Request my_request; 
  //std::cerr << "before send/recv init: " << typeid(a.data()).name() << ", " << sizeof(a.data())*a.size() << std::endl;
  //std::cerr << "sizes: " << &a[a.size()-1] - &a[0] << std::endl;
  if (0 == rank) {
    int dst = 1;
    Kokkos::parallel_for(
    a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; }
			 );
    KokkosComm::Experimental::stream::send(a, 1, 0, MPI_COMM_WORLD, ctx._mem_info, &my_request);
  }
  else if (1 == rank){
    KokkosComm::Experimental::stream::recv(b, 0, 0, MPI_COMM_WORLD, ctx._mem_info, &my_request);
  }
  MPIS_Match(&my_request, MPI_STATUS_IGNORE);
  //hipStreamSynchronize((hipStream_t) _my_stream);  
  
  MPIS_Enqueue_startall( ctx._my_queue, 1, &my_request );
  //hipStreamSynchronize((hipStream_t) _my_stream);
  MPIS_Enqueue_waitall( ctx._my_queue );
  std::cerr << " after start all" << std::endl;

  if( 1 == rank) {
    int src = 0; int errs;
    Kokkos::parallel_reduce(
			    b.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += b(i) != i; }, errs);     
    ASSERT_EQ(errs, 0);
    std::cerr << "errs: " << errs << std::endl;
  }

  //hipStreamSynchronize((hipStream_t) _my_stream);
  MPIS_Request_free(&my_request);
/*  MPIS_Queue_free( &_my_queue );
    MPI_Info_free( &_mem_info );*/
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

  MPI_Info _mem_info;
  MPI_Info_create( &_mem_info );
  MPI_Info_set(_mem_info, "mpi_memory_alloc_kinds", "rocm:device:coarse");

  void* _my_stream;
  MPIS_Queue _my_queue;

  if constexpr ( std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::HIP>){
		 _my_stream = Kokkos::HIP().hip_stream();
  }
  else{
    _my_stream = nullptr;
  }
  MPIS_Queue_init(&_my_queue, CXI, &_my_stream);     

  // set up send and recv
  MPIS_Request my_request; 
  //std::cerr << "before send/recv init: " << typeid(a.data()).name() << ", " << sizeof(a.data())*a.size() << std::endl;
  //std::cerr << "sizes: " << &a[a.size()-1] - &a[0] << std::endl;
  if (0 == rank) {
    int dst = 1;
    Kokkos::parallel_for(
    a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; }
			 );
    KokkosComm::Experimental::stream::send(Kokkos::DefaultExecutionSpace(), a, 1, 0, MPI_COMM_WORLD, _mem_info, &my_request);
  }
  else if (1 == rank){
    KokkosComm::Experimental::stream::recv(Kokkos::DefaultExecutionSpace(), b, 0, 0, MPI_COMM_WORLD, _mem_info, &my_request);
  }
  MPIS_Match(&my_request, MPI_STATUS_IGNORE);
  //hipStreamSynchronize((hipStream_t) _my_stream);  
  
  MPIS_Enqueue_startall( _my_queue, 1, &my_request );
  //hipStreamSynchronize((hipStream_t) _my_stream);
  MPIS_Enqueue_waitall( _my_queue );
  std::cerr << " after start all" << std::endl;

  if( 1 == rank) {
    int src = 0; int errs;
    Kokkos::parallel_reduce(
			    b.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += b(i) != i; }, errs);     
    ASSERT_EQ(errs, 0);
    std::cerr << "errs: " << errs << std::endl;
  }

  //hipStreamSynchronize((hipStream_t) _my_stream);
  MPIS_Request_free(&my_request);
  MPIS_Queue_free( &_my_queue );
  MPI_Info_free( &_mem_info );
}
  
TYPED_TEST(MpiSendRecv, 1D_contig_standard) {
  send_comm_mode_1d_contig<CommModeStandard, typename TestFixture::Scalar>();
}

TYPED_TEST(MpiSendRecv, 1D_contig_ready) {
  send_comm_mode_1d_contig<CommModeReady, typename TestFixture::Scalar>();
}

  //TYPED_TEST(MpiSendRecv, 1D_noncontig_standard) {
  //send_comm_mode_1d_noncontig<CommModeStandard, typename TestFixture::Scalar>();
  //}

}  // namespace
