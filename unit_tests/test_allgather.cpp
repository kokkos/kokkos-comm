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
class AllGatherFixture : public testing::Test {
 public:
  using Scalar = T;
};

template <KokkosComm::CommunicationSpace CommSpace, typename Scalar>
auto allgather_0d() -> void {
  auto comm      = test_utils::make_communicator<CommSpace>();
  auto exec      = comm.exec();
  const int size = comm.size();
  const int rank = comm.rank();

  Kokkos::View<Scalar> sv("sv");
  Kokkos::View<Scalar*> rv("rv", size);

  // Prepare send view, 1 element per sender: their rank
  Kokkos::parallel_for(
      Kokkos::RangePolicy(exec, 0, sv.extent(0)), KOKKOS_LAMBDA(const int) { sv() = rank; }
  );
  exec.fence();

  KokkosComm::Experimental::allgather(comm, sv, rv).wait();

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(const int src, int& lsum) { lsum += rv(src) != src; }, errs
  );
  EXPECT_EQ(errs, 0);
}

template <KokkosComm::CommunicationSpace CommSpace, typename Scalar>
auto allgather_contig_1d() -> void {
  auto comm      = test_utils::make_communicator<CommSpace>();
  auto exec      = comm.exec();
  const int size = comm.size();
  const int rank = comm.rank();

  const int n_contrib = 100;
  Kokkos::View<Scalar*> sv("sv", n_contrib);
  Kokkos::View<Scalar*> rv("rv", size * n_contrib);

  // Prepare send view
  Kokkos::parallel_for(
      Kokkos::RangePolicy(exec, 0, sv.extent(0)), KOKKOS_LAMBDA(const int i) { sv(i) = rank + i; }
  );
  exec.fence();

  KokkosComm::Experimental::allgather(comm, sv, rv).wait();

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(const int i, int& lsum) {
        const int src = i / n_contrib;
        const int j   = i % n_contrib;
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

#define KC_DEFINE_ALLGATHER_TESTS(SuiteName, TypeList, CommSpace)                        \
  template <typename T>                                                                  \
  class SuiteName : public AllGatherFixture<T> {};                                       \
                                                                                         \
  TYPED_TEST_SUITE(SuiteName, TypeList);                                                 \
                                                                                         \
  TYPED_TEST(SuiteName, 0D) { allgather_0d<CommSpace, typename TestFixture::Scalar>(); } \
  TYPED_TEST(SuiteName, Contiguous1D) { allgather_contig_1d<CommSpace, typename TestFixture::Scalar>(); }

#if defined(KOKKOSCOMM_ENABLE_MPI)
KC_DEFINE_ALLGATHER_TESTS(MpiAllGather, MpiScalarTypes, KokkosComm::MpiSpace)
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL)
KC_DEFINE_ALLGATHER_TESTS(NcclAllGather, XcclScalarTypes, KokkosComm::Experimental::NcclSpace)
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
KC_DEFINE_ALLGATHER_TESTS(RcclAllGather, XcclScalarTypes, KokkosComm::Experimental::RcclSpace)
#endif

#undef KC_DEFINE_ALLGATHER_TESTS

}  // namespace
