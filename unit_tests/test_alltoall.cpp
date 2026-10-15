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
class AllToAllFixture : public testing::Test {
 public:
  using Scalar = T;
};

template <KokkosComm::CommunicationSpace CommSpace, typename Scalar>
auto alltoall_contig_1d() -> void {
  auto comm      = test_utils::make_communicator<CommSpace>();
  auto exec      = comm.exec();
  const int size = comm.size();
  const int rank = comm.rank();

  int n_contrib = 100;
  Kokkos::View<Scalar*> sv("sv", size * n_contrib);
  Kokkos::View<Scalar*> rv("rv", size * n_contrib);

  // Prepare send view
  Kokkos::parallel_for(
      Kokkos::RangePolicy(exec, 0, sv.extent(0)), KOKKOS_LAMBDA(int i) { sv(i) = rank + i; }
  );
  exec.fence();

  // Perform all-to-all operation
  KokkosComm::Experimental::alltoall(comm, sv, rv, n_contrib).wait();

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(const int i, int& lsum) {
        const int src = i / n_contrib;                       // who sent this data
        const int j   = rank * n_contrib + (i % n_contrib);  // what index i was at the source
        lsum += rv(i) != src + j;
      },
      errs
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

#define KC_DEFINE_ALLTOALL_TESTS(SuiteName, TypeList, CommSpace) \
  template <typename T>                                          \
  class SuiteName : public AllToAllFixture<T> {};                \
                                                                 \
  TYPED_TEST_SUITE(SuiteName, TypeList);                         \
                                                                 \
  TYPED_TEST(SuiteName, Contiguous1D) { alltoall_contig_1d<CommSpace, typename TestFixture::Scalar>(); }

#if defined(KOKKOSCOMM_ENABLE_MPI)
KC_DEFINE_ALLTOALL_TESTS(MpiAllToAll, MpiScalarTypes, KokkosComm::MpiSpace)
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL)
KC_DEFINE_ALLTOALL_TESTS(NcclAllToAll, XcclScalarTypes, KokkosComm::Experimental::NcclSpace)
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
KC_DEFINE_ALLTOALL_TESTS(RcclAllToAll, XcclScalarTypes, KokkosComm::Experimental::RcclSpace)
#endif

#undef KC_DEFINE_ALLTOALL_TESTS

}  // namespace
