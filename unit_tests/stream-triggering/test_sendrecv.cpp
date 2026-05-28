// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <type_traits>

#include <gtest/gtest.h>

#include <stream-triggering.h>

#include <KokkosComm/KokkosComm.hpp>

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

  Kokkos::View<Scalar *> a("a", 1000);
  MPIS_Hello_world();

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }

  if (0 == rank) {
    int dst = 1;
    Kokkos::parallel_for(
        a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; }
    );
    KokkosComm::Experimental::stream::send(Kokkos::DefaultExecutionSpace(), a, dst, 0, MPI_COMM_WORLD);
  }
  else if (1 == rank) {
    int src = 0;
    KokkosComm::Experimental::stream::recv(Kokkos::DefaultExecutionSpace(), a, src, 0, MPI_COMM_WORLD);
    int errs;
    Kokkos::parallel_reduce(
			    a.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += a(i) != i; }, errs);
    ASSERT_EQ(errs, 0);
  }
  /*
    KokkosComm::Experimental::stream::send(Kokkos::DefaultExecutionSpace(), a, dst, 0, MPI_COMM_WORLD, SendMode{});
  } else if (1 == rank) {
    int src = 0;
    KokkosComm::Experimental::stream::recv(Kokkos::DefaultExecutionSpace(), a, src, 0, MPI_COMM_WORLD);
    int errs;
    Kokkos::parallel_reduce(
        a.extent(0), KOKKOS_LAMBDA(const int &i, int &lsum) { lsum += a(i) != i; }, errs
    );
    ASSERT_EQ(errs, 0);
    }*/
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
