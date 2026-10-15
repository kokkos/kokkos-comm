// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include <KokkosComm/KokkosComm.hpp>

#include "communication_space_utils.hpp"
#include "view_utils.hpp"

namespace {

template <KokkosComm::CommunicationSpace CommSpace, KokkosComm::KokkosView View>
void test_core_send_recv(const View& v) {
  auto comm      = test_utils::make_communicator<CommSpace>();
  auto exec      = comm.exec();
  const int size = comm.size();
  const int rank = comm.rank();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  const int src = 0;
  const int dst = 1;

  if (rank == src) {
    test_utils::init_view(exec, v);
    exec.fence();
    KokkosComm::send(comm, v, dst).wait();
  } else if (rank == dst) {
    KokkosComm::recv(comm, v, src).wait();
    int errs = test_utils::count_errors(v);
    ASSERT_EQ(errs, 0);
  }
}

template <typename T>
class SendRecvFixture : public ::testing::Test {
 public:
  using Scalar = T;
};

using MpiScalarTypes =
    ::testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
using XcclScalarTypes = ::testing::Types<float, double, int, int64_t>;

#define KC_DEFINE_SEND_RECV_TESTS(SuiteName, TypeList, CommSpace)                \
  template <typename T>                                                          \
  class SuiteName : public SendRecvFixture<T> {};                                \
                                                                                 \
  TYPED_TEST_SUITE(SuiteName, TypeList);                                         \
                                                                                 \
  TYPED_TEST(SuiteName, Contig1D) {                                              \
    auto v = test_utils::build_view<typename TestFixture::Scalar, 1>(            \
        test_utils::get_exec<CommSpace>(), test_utils::Contig{}, "v", 1013       \
    );                                                                           \
    test_core_send_recv<CommSpace>(v);                                           \
  }                                                                              \
                                                                                 \
  TYPED_TEST(SuiteName, NonContig1D) {                                           \
    auto v = test_utils::build_view<typename TestFixture::Scalar, 1>(            \
        test_utils::get_exec<CommSpace>(), test_utils::NonContig{}, "v", 1013    \
    );                                                                           \
    test_core_send_recv<CommSpace>(v);                                           \
  }                                                                              \
                                                                                 \
  TYPED_TEST(SuiteName, Contig2D) {                                              \
    auto v = test_utils::build_view<typename TestFixture::Scalar, 2>(            \
        test_utils::get_exec<CommSpace>(), test_utils::Contig{}, "v", 137, 17    \
    );                                                                           \
    test_core_send_recv<CommSpace>(v);                                           \
  }                                                                              \
                                                                                 \
  TYPED_TEST(SuiteName, NonContig2D) {                                           \
    auto v = test_utils::build_view<typename TestFixture::Scalar, 2>(            \
        test_utils::get_exec<CommSpace>(), test_utils::NonContig{}, "v", 137, 17 \
    );                                                                           \
    test_core_send_recv<CommSpace>(v);                                           \
  }

#if defined(KOKKOSCOMM_ENABLE_MPI)
KC_DEFINE_SEND_RECV_TESTS(MpiSendRecv, MpiScalarTypes, KokkosComm::MpiSpace)
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL)
KC_DEFINE_SEND_RECV_TESTS(NcclSendRecv, XcclScalarTypes, KokkosComm::Experimental::NcclSpace)
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
KC_DEFINE_SEND_RECV_TESTS(RcclSendRecv, XcclScalarTypes, KokkosComm::Experimental::RcclSpace)
#endif

#undef KC_DEFINE_SEND_RECV_TESTS

}  // namespace
