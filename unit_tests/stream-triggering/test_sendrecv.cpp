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
  using namespace KokkosComm::mpi;

template <typename T>
class MpiSendRecv : public testing::Test {
 public:
  using Scalar = T;
};

  using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(MpiSendRecv, ScalarTypes);

  //template <CommunicationMode SendMode, typename Scalar>
  template <typename Scalar>
void send_comm_mode_1d_contig() {
  //if constexpr (std::is_same_v<SendMode, CommModeReady>) {
  //  GTEST_SKIP() << "Skipping test for ready-mode send";
  //}
      int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  Kokkos::View<Scalar *> a("a", 1000); // send
  Kokkos::View<Scalar *> b("b", 1000); // recv

  // size_t get_size(View)
  // returns it in bytes
  size_t underlying_size;
  if constexpr(std::is_same_v<Scalar, Kokkos::complex<float>> == true || std::is_same_v<Scalar, Kokkos::complex<double>> == true){
      underlying_size = 2*sizeof(b.data())*a.size();
    }
  else {
    underlying_size = sizeof(b.data())*a.size();
  }
  // do same but with MPI_Datatype as added feature

  MPI_Info _mem_info;
  MPI_Info_create( &_mem_info );
  MPI_Info_set(_mem_info, "mpi_memory_alloc_kinds", "rocm:device:fine");

  void* _my_stream;
  MPIS_Queue _my_queue;

  //auto stream_ctx = test_utils::mpi_advance::Ctx(Kokkos::DefaultExecutionSpace());
  if constexpr ( std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::HIP>)
		 {
		 std::cerr << "hip stream" << std::endl;
		 _my_stream = Kokkos::HIP().hip_stream();
}
  else{
    std::cerr << "not hip stream" << std::endl;
    _my_stream = nullptr;
  }
  MPIS_Queue_init(&_my_queue, CXI, &_my_stream);     

  // set up send and recv
  MPIS_Request my_reqs; // 0 send; 1 recv
  std::cerr << "before send/recv init: " << typeid(a.data()).name() << ", " << sizeof(a.data())*a.size() << std::endl;
  std::cerr << "sizes: " << &a[a.size()-1] - &a[0] << std::endl;
  if (0 == rank) {
    // send
    // send -> send(exec, a, dst, tag, comm, sendmode, ctx)
    // set up requests and others within ctx, if send then 1 request. Set tag in ctx
    // check if contig
    MPIS_Send_init(a.data(), underlying_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD, _mem_info,
                       &my_reqs);
  }
  else if (1 == rank){
    MPIS_Recv_init(b.data(), underlying_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, _mem_info, &my_reqs);
    }
  MPIS_Match(&my_reqs, MPI_STATUS_IGNORE);
  if(rank == 1){
    sleep(5);}
   hipStreamSynchronize((hipStream_t) _my_stream);  
  MPI_Barrier(MPI_COMM_WORLD);
  if (0 == rank) {
    int dst = 1;
    Kokkos::parallel_for(
    a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; }
			 );
    //KokkosComm::Experimental::stream::send(Kokkos::DefaultExecutionSpace(), a, dst, 0, MPI_COMM_WORLD);
    
  }
  else if (1 == rank) {
    int src = 0;
    //KokkosComm::Experimental::stream::recv(Kokkos::DefaultExecutionSpace(), a, src, 0, MPI_COMM_WORLD);
    int errs;
    //Kokkos::parallel_reduce(
    //			    b.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += b(i) != i; }, errs);
    //ASSERT_EQ(errs, 0);
  }
  //std::cerr << " before start all" << std::endl;
  MPIS_Enqueue_startall( _my_queue, 1, &my_reqs );
  hipStreamSynchronize((hipStream_t) _my_stream);
  MPIS_Enqueue_waitall( _my_queue );
  std::cerr << " after start all" << std::endl;
   if( 1 == rank) {
     int src = 0; int errs;
     Kokkos::parallel_reduce(
     b.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += b(i) != i; }, errs);     
      //ASSERT_EQ(errs, 0);
      std::cerr << "errs: " << errs << std::endl;
   }
   //std::cerr << "Before frees!" << std::endl;
  // free everything
   hipStreamSynchronize((hipStream_t) _my_stream);
   MPIS_Request_free(&my_reqs);
  MPIS_Queue_free( &_my_queue );
  MPI_Info_free( &_mem_info );
  std::cerr << "Freed!" << std::endl;  
}

TYPED_TEST(MpiSendRecv, 1D_contig_standard) {
  //send_comm_mode_1d_contig<CommModeStandard, typename TestFixture::Scalar>();
  send_comm_mode_1d_contig<typename TestFixture::Scalar>();
}

  TYPED_TEST(MpiSendRecv, 1D_contig_ready) { //send_comm_mode_1d_contig<CommModeReady, typename TestFixture::Scalar>();
  send_comm_mode_1d_contig<typename TestFixture::Scalar>();
  }

TYPED_TEST(MpiSendRecv, 1D_contig_synchronous) {
  //send_comm_mode_1d_contig<CommModeSynchronous, typename TestFixture::Scalar>();
  send_comm_mode_1d_contig<typename TestFixture::Scalar>();
}

}  // namespace
