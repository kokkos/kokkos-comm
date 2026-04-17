// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <type_traits>

#include <KokkosComm/KokkosComm.hpp>

namespace {

using namespace KokkosComm::mpi;

template <typename T>
class ChannelSendRecv : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(ChannelSendRecv, ScalarTypes);

template <typename Scalar>
void test_channel_hostspace() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  const int dest_rank = (rank + 1) % size;         // send to next rank
  const int src_rank  = (rank - 1 + size) % size;  // recv from prev rank
  const int tag       = 42;

  KokkosComm::Channel<> channel(dest_rank, src_rank, tag, MPI_COMM_WORLD);

  const int N = 10;

  // Create host views
  Kokkos::View<Scalar*, Kokkos::HostSpace> send_host("send_host", N);
  Kokkos::View<Scalar*, Kokkos::HostSpace> recv_host("recv_host", N);

  for (int i = 0; i < N; i++) send_host(i) = static_cast<Scalar>(rank * N + i);

  channel.sendinit(send_host);
  channel.recvinit(recv_host);

  channel.start();
  channel.wait();

  int errs = 0;
  for (int i = 0; i < N; i++) {
    const Scalar expected = static_cast<Scalar>(src_rank * N + i);
    if (recv_host(i) != expected) {
      errs++;
    }
  }
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
void test_channel_execspace() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  const int dest_rank = (rank + 1) % size;         // send to next rank
  const int src_rank  = (rank - 1 + size) % size;  // recv from prev rank
  const int tag       = 42;

  KokkosComm::Channel<> channel(dest_rank, src_rank, tag, MPI_COMM_WORLD);

  const int N = 10;

  // Create host view
  Kokkos::View<Scalar*, Kokkos::HostSpace> recv_host("recv_host", N);
  // Create device views
  Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace> send_dev("send_dev", N);
  Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace> recv_dev("recv_dev", N);

  GTEST_LOG_(INFO) << "Kokkos Execution Space: " << Kokkos::DefaultExecutionSpace::name();
  Kokkos::parallel_for(
      "init_send_dev", N, KOKKOS_LAMBDA(int i) { send_dev(i) = static_cast<Scalar>(rank * N + i); }
  );
  Kokkos::fence();

  channel.sendinit(send_dev);
  channel.recvinit(recv_dev);

  channel.start();
  channel.wait();

  Kokkos::deep_copy(recv_host, recv_dev);

  int errs = 0;
  for (int i = 0; i < N; i++) {
    const Scalar expected = static_cast<Scalar>(src_rank * N + i);
    if (recv_host(i) != expected) {
      errs++;
    }
  }
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
void test_channel_reuse() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  const int dest_rank = (rank + 1) % size;         // send to next rank
  const int src_rank  = (rank - 1 + size) % size;  // recv from prev rank
  const int tag       = 42;

  KokkosComm::Channel<> channel(dest_rank, src_rank, tag, MPI_COMM_WORLD);

  const int N = 10;

  // Create host view
  Kokkos::View<Scalar*, Kokkos::HostSpace> recv_host("recv_host", N);
  // Create device views
  Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace> send_dev("send_dev", N);
  Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace> recv_dev("recv_dev", N);

  GTEST_LOG_(INFO) << "Kokkos Execution Space: " << Kokkos::DefaultExecutionSpace::name();
  int errs = 0;
  for (int i = 0; i < 3; i++) {
    Kokkos::parallel_for(
        "init_send_dev", N, KOKKOS_LAMBDA(int i) { send_dev(i) = static_cast<Scalar>(rank * N + i); }
    );
    Kokkos::fence();

    channel.sendinit(send_dev);
    channel.recvinit(recv_dev);

    channel.start();
    channel.wait();

    Kokkos::deep_copy(recv_host, recv_dev);

    for (int j = 0; j < N; j++) {
      const Scalar expected = static_cast<Scalar>(src_rank * N + j);
      if (recv_host(j) != expected) {
        errs++;
      }
    }
  }
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(ChannelSendRecv, 1D_contig_sendrecv_hostspace) { test_channel_hostspace<typename TestFixture::Scalar>(); }
TYPED_TEST(ChannelSendRecv, 1D_contig_sendrecv_execspace) { test_channel_execspace<typename TestFixture::Scalar>(); }
TYPED_TEST(ChannelSendRecv, 1D_contig_sendrecv_reuse) { test_channel_reuse<typename TestFixture::Scalar>(); }

}  // namespace
