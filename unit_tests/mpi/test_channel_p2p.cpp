// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <type_traits>

#include <gtest/gtest.h>
#include <KokkosComm/KokkosComm.hpp>

namespace {

using namespace KokkosComm::mpi;
using ChannelType = KokkosComm::Channel<KokkosComm::MpiSpace>;

static_assert(std::is_same_v<KokkosComm::Channel<>, ChannelType>);
static_assert(!std::is_copy_constructible_v<ChannelType>);
static_assert(!std::is_copy_assignable_v<ChannelType>);
static_assert(std::is_nothrow_move_constructible_v<ChannelType>);
static_assert(std::is_nothrow_move_assignable_v<ChannelType>);

template <typename T>
class ChannelP2P : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(ChannelP2P, ScalarTypes);

template <typename Scalar, typename ExecutionSpace>
void test_channel_p2p() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  const int dst_rank = (rank + 1) % size;         // send to next rank
  const int src_rank = (rank - 1 + size) % size;  // recv from prev rank
  const int tag      = 42;
  const int N        = 10;

  using memory_space = typename ExecutionSpace::memory_space;
  Kokkos::View<Scalar*, memory_space> send("send", N);
  Kokkos::View<Scalar*, memory_space> recv("recv", N);

  ChannelType channel(dst_rank, src_rank, tag, MPI_COMM_WORLD);

  ExecutionSpace exec;
  GTEST_LOG_(INFO) << "Kokkos Execution Space: " << ExecutionSpace::name();
  Kokkos::parallel_for(
      "init_send", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, N),
      KOKKOS_LAMBDA(int i) { send(i) = static_cast<Scalar>(rank * N + i); }
  );
  exec.fence();

  channel.sendinit(send);
  channel.recvinit(recv);

  channel.start();
  channel.wait();

  auto recv_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv);

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
void test_channel_p2p() {
  using host_execution_space    = Kokkos::DefaultHostExecutionSpace;
  using default_execution_space = Kokkos::DefaultExecutionSpace;

  test_channel_p2p<Scalar, host_execution_space>();
  if constexpr (!std::is_same_v<
                    typename host_execution_space::memory_space, typename default_execution_space::memory_space>) {
    test_channel_p2p<Scalar, default_execution_space>();
  }
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

  const int N = 10;

  // Create host view
  Kokkos::View<Scalar*, Kokkos::HostSpace> recv_host("recv_host", N);
  // Create device views
  Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace> send_dev("send_dev", N);
  Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace> recv_dev("recv_dev", N);

  ChannelType channel(dest_rank, src_rank, tag, MPI_COMM_WORLD);

  channel.sendinit(send_dev);
  channel.recvinit(recv_dev);

  GTEST_LOG_(INFO) << "Kokkos Execution Space: " << Kokkos::DefaultExecutionSpace::name();
  int errs = 0;
  for (int cycle = 0; cycle < 3; ++cycle) {
    Kokkos::deep_copy(recv_dev, static_cast<Scalar>(-1));
    Kokkos::parallel_for(
        "init_send_dev", N,
        KOKKOS_LAMBDA(int i) { send_dev(i) = static_cast<Scalar>((cycle + 1) * size * N + rank * N + i); }
    );
    Kokkos::fence();

    channel.start();
    channel.wait();

    Kokkos::deep_copy(recv_host, recv_dev);

    for (int j = 0; j < N; j++) {
      const Scalar expected = static_cast<Scalar>((cycle + 1) * size * N + src_rank * N + j);
      if (recv_host(j) != expected) {
        errs++;
      }
    }
  }
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
void test_channel_move_construction() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  const int dest_rank = (rank + 1) % size;
  const int src_rank  = (rank - 1 + size) % size;
  const int tag       = 43;
  const int N         = 10;

  Kokkos::View<Scalar*, Kokkos::HostSpace> send_host("move_construct_send", N);
  Kokkos::View<Scalar*, Kokkos::HostSpace> recv_host("move_construct_recv", N);

  ChannelType source(dest_rank, src_rank, tag, MPI_COMM_WORLD);
  source.sendinit(send_host);
  source.recvinit(recv_host);

  ChannelType destination(std::move(source));

  int errs = 0;
  for (int cycle = 0; cycle < 2; ++cycle) {
    for (int i = 0; i < N; ++i) {
      send_host(i) = static_cast<Scalar>((cycle + 1) * size * N + rank * N + i);
      recv_host(i) = static_cast<Scalar>(-1);
    }

    destination.start();
    destination.wait();

    for (int i = 0; i < N; ++i) {
      const Scalar expected = static_cast<Scalar>((cycle + 1) * size * N + src_rank * N + i);
      if (recv_host(i) != expected) {
        ++errs;
      }
    }
  }
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
void test_channel_move_assignment() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  const int dest_rank = (rank + 1) % size;
  const int src_rank  = (rank - 1 + size) % size;
  const int N         = 10;

  Kokkos::View<Scalar*, Kokkos::HostSpace> source_send("move_assign_source_send", N);
  Kokkos::View<Scalar*, Kokkos::HostSpace> source_recv("move_assign_source_recv", N);
  Kokkos::View<Scalar*, Kokkos::HostSpace> destination_send("move_assign_destination_send", N);
  Kokkos::View<Scalar*, Kokkos::HostSpace> destination_recv("move_assign_destination_recv", N);

  ChannelType source(dest_rank, src_rank, 44, MPI_COMM_WORLD);
  source.sendinit(source_send);
  source.recvinit(source_recv);

  ChannelType destination(dest_rank, src_rank, 45, MPI_COMM_WORLD);
  destination.sendinit(destination_send);
  destination.recvinit(destination_recv);

  destination = std::move(source);

  int errs = 0;
  for (int cycle = 0; cycle < 2; ++cycle) {
    for (int i = 0; i < N; ++i) {
      source_send(i) = static_cast<Scalar>((cycle + 1) * size * N + rank * N + i);
      source_recv(i) = static_cast<Scalar>(-1);
    }

    destination.start();
    destination.wait();

    for (int i = 0; i < N; ++i) {
      const Scalar expected = static_cast<Scalar>((cycle + 1) * size * N + src_rank * N + i);
      if (source_recv(i) != expected) {
        ++errs;
      }
    }
  }
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(ChannelP2P, 1D_contig) { test_channel_p2p<typename TestFixture::Scalar>(); }
TYPED_TEST(ChannelP2P, 1D_contig_reuse) { test_channel_reuse<typename TestFixture::Scalar>(); }
TYPED_TEST(ChannelP2P, registered_requests_survive_move_construction) {
  test_channel_move_construction<typename TestFixture::Scalar>();
}
TYPED_TEST(ChannelP2P, registered_requests_survive_move_assignment) {
  test_channel_move_assignment<typename TestFixture::Scalar>();
}

}  // namespace
