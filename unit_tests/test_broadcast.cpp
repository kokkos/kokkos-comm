// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#include "communication_space_utils.hpp"

namespace {

template <typename T>
class BroadcastFixture : public testing::Test {
 public:
  using Scalar = T;
};

template <KokkosComm::CommunicationSpace CommSpace, typename Scalar>
auto broadcast_0d() -> void {
  auto comm      = test_utils::make_communicator<CommSpace>();
  auto exec      = comm.exec();
  const int size = comm.size();
  const int rank = comm.rank();
  const int root = 0;

  Kokkos::View<Scalar> v("v");

  if (rank == root) {
    // Prepare broadcast view
    Kokkos::parallel_for(
        Kokkos::RangePolicy(exec, 0, v.extent(0)), KOKKOS_LAMBDA(const int) { v() = size; }
    );
    exec.fence();
  }

  KokkosComm::Experimental::broadcast(comm, v, root).wait();

  int errs;
  Kokkos::parallel_reduce(
      v.extent(0), KOKKOS_LAMBDA(const int, int& lsum) { lsum += v() != size; }, errs
  );
  EXPECT_EQ(errs, 0);
}

template <KokkosComm::CommunicationSpace CommSpace, typename Scalar>
auto broadcast_contig_1d() -> void {
  auto comm      = test_utils::make_communicator<CommSpace>();
  auto exec      = comm.exec();
  const int size = comm.size();
  const int rank = comm.rank();
  const int root = 0;

  Kokkos::View<Scalar*> v("v", 100);

  if (rank == root) {
    // Prepare broadcast view
    Kokkos::parallel_for(
        Kokkos::RangePolicy(exec, 0, v.extent(0)), KOKKOS_LAMBDA(const int i) { v(i) = size + i; }
    );
    exec.fence();
  }

  KokkosComm::Experimental::broadcast(comm, v, root).wait();

  int errs;
  Kokkos::parallel_reduce(
      v.extent(0), KOKKOS_LAMBDA(const int i, int& lsum) { lsum += (v(i) != size + i); }, errs
  );
  EXPECT_EQ(errs, 0);
}

#if defined(KOKKOSCOMM_ENABLE_MPI)
using MpiScalarTypes =
    testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
#endif
#if defined(KOKKOSCOMM_ENABLE_NCCL) || defined(KOKKOSCOMM_ENABLE_RCCL)
using XcclScalarTypes = testing::Types<float, double, int, int64_t>;
#endif

#define KC_DEFINE_BROADCAST_TESTS(SuiteName, TypeList, CommSpace)                        \
  template <typename T>                                                                  \
  class SuiteName : public BroadcastFixture<T> {};                                       \
                                                                                         \
  TYPED_TEST_SUITE(SuiteName, TypeList);                                                 \
                                                                                         \
  TYPED_TEST(SuiteName, 0D) { broadcast_0d<CommSpace, typename TestFixture::Scalar>(); } \
  TYPED_TEST(SuiteName, Contiguous1D) { broadcast_contig_1d<CommSpace, typename TestFixture::Scalar>(); }

#if defined(KOKKOSCOMM_ENABLE_MPI)
KC_DEFINE_BROADCAST_TESTS(MpiBroadcast, MpiScalarTypes, KokkosComm::MpiSpace)
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL)
KC_DEFINE_BROADCAST_TESTS(NcclBroadcast, XcclScalarTypes, KokkosComm::Experimental::NcclSpace)
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
KC_DEFINE_BROADCAST_TESTS(RcclBroadcast, XcclScalarTypes, KokkosComm::Experimental::RcclSpace)
#endif

#undef KC_DEFINE_BROADCAST_TESTS

}  // namespace
