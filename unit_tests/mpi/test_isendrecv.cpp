// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core_fwd.hpp>
#include <type_traits>

#include <gtest/gtest.h>

#include <KokkosComm/KokkosComm.hpp>

namespace {

using Ex = Kokkos::DefaultExecutionSpace;
using Co = KokkosComm::MpiSpace;

using namespace KokkosComm::mpi;

template <typename T>
class IsendRecv : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes =
    ::testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
TYPED_TEST_SUITE(IsendRecv, ScalarTypes);

template <CommunicationMode IsendMode, typename Scalar>
void isend_comm_mode_1d_contig() {
  if constexpr (std::is_same_v<IsendMode, CommModeReady>) {
    GTEST_SKIP() << "Skipping test for ready-mode send";
  }

  Kokkos::View<Scalar*> a("a", 1000);

  auto h = KokkosComm::Communicator<>::from_raw(MPI_COMM_WORLD);
  if (h.size() < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << h.size() << " provided)";
  }

  if (0 == h.rank()) {
    int dst = 1;
    Kokkos::parallel_for(
        a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; }
    );
    KokkosComm::mpi::isend(h, a, dst, 0, IsendMode{}).value().wait();
  } else if (1 == h.rank()) {
    int src = 0;
    KokkosComm::mpi::recv(h.exec(), a, src, 0, h.comm());
    int errs;
    Kokkos::parallel_reduce(
        a.extent(0), KOKKOS_LAMBDA(const int& i, int& lsum) { lsum += a(i) != Scalar(i); }, errs
    );
    ASSERT_EQ(errs, 0);
  }
}

template <CommunicationMode IsendMode, typename Scalar>
void isend_comm_mode_1d_noncontig() {
  if constexpr (std::is_same_v<IsendMode, CommModeReady>) {
    GTEST_SKIP() << "Skipping test for ready-mode send";
  }

  // this is C-style layout, i.e. b(0,0) is next to b(0,1)
  Kokkos::View<Scalar**, Kokkos::LayoutRight> b("a", 10, 10);
  auto a = Kokkos::subview(b, Kokkos::ALL, 2);  // take column 2 (non-contiguous)

  auto h = KokkosComm::Communicator<>::from_raw(MPI_COMM_WORLD);
  if (h.size() < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << h.size() << " provided)";
  }

  if (0 == h.rank()) {
    int dst = 1;
    Kokkos::parallel_for(
        a.extent(0), KOKKOS_LAMBDA(const int i) { a(i) = i; }
    );
    KokkosComm::mpi::isend(h, a, dst, 0, IsendMode{}).value().wait();
  } else if (1 == h.rank()) {
    int src = 0;
    KokkosComm::mpi::recv(h.exec(), a, src, 0, h.comm());
    int errs;
    Kokkos::parallel_reduce(
        a.extent(0), KOKKOS_LAMBDA(const int& i, int& lsum) { lsum += a(i) != Scalar(i); }, errs
    );
    ASSERT_EQ(errs, 0);
  }
}

TYPED_TEST(IsendRecv, 1D_contig_standard) {
  isend_comm_mode_1d_contig<CommModeStandard, typename TestFixture::Scalar>();
}

TYPED_TEST(IsendRecv, 1D_contig_ready) { isend_comm_mode_1d_contig<CommModeReady, typename TestFixture::Scalar>(); }

TYPED_TEST(IsendRecv, 1D_contig_synchronous) {
  isend_comm_mode_1d_contig<CommModeSynchronous, typename TestFixture::Scalar>();
}

TYPED_TEST(IsendRecv, 1D_noncontig_standard) {
  isend_comm_mode_1d_noncontig<CommModeStandard, typename TestFixture::Scalar>();
}

TYPED_TEST(IsendRecv, 1D_noncontig_ready) {
  isend_comm_mode_1d_noncontig<CommModeReady, typename TestFixture::Scalar>();
}

TYPED_TEST(IsendRecv, 1D_noncontig_synchronous) {
  isend_comm_mode_1d_noncontig<CommModeSynchronous, typename TestFixture::Scalar>();
}

TEST(MpiIsendRecvError, NullBuffer) {
  auto h = KokkosComm::Communicator<>::from_raw(MPI_COMM_WORLD);
  if (h.size() < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << h.size() << " provided)";
  }

  // Unmanaged, contiguous view: null data pointer but non-zero extent
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v(
      static_cast<int*>(nullptr), 1013
  );

  if (0 == h.rank()) {
    auto req = KokkosComm::mpi::isend(h, v, 1, 0);
    ASSERT_FALSE(req.has_value());
    EXPECT_EQ(req.error().code, KokkosComm::ErrorCode::MPIError);
    EXPECT_EQ(req.error().backend_code, MPI_ERR_BUFFER);
  } else if (1 == h.rank()) {
    MPI_Request mpi_req;
    auto status = KokkosComm::mpi::recv(h.exec(), v, 0, 0, h.comm());
    ASSERT_FALSE(status.has_value());
    EXPECT_EQ(status.error().code, KokkosComm::ErrorCode::MPIError);
    EXPECT_EQ(status.error().backend_code, MPI_ERR_BUFFER);
  }
}

}  // namespace
