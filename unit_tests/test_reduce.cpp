// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#include "communication_space_utils.hpp"

namespace {

template <typename T>
class ReduceFixture : public testing::Test {
 public:
  using Scalar = T;
};

/// Each rank fills its sendbuf[i] with `rank + i`
/// operation is sum, so recvbuf[i] should be sum(0..size) + i * size
template <KokkosComm::CommunicationSpace CommSpace, typename Scalar>
auto reduce_contig_1d() -> void {
// FIXME_EXTERNAL #215
#if defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
  if constexpr (std::is_same_v<CommSpace, KokkosComm::MpiSpace>) {
    GTEST_SKIP() << "Unimplemented test for Open MPI + CUDA/HIP";
  }
#endif
  auto comm      = test_utils::make_communicator<CommSpace>();
  auto exec      = comm.exec();
  const int size = comm.size();
  const int rank = comm.rank();
  const int root = 0;

  const int n_contrib = 10;
  Kokkos::View<Scalar*> sv("sv", n_contrib);
  Kokkos::View<Scalar*> rv("rv", n_contrib);

  // Prepare send buffer
  Kokkos::parallel_for(
      Kokkos::RangePolicy(exec, 0, sv.extent(0)), KOKKOS_LAMBDA(const int i) { sv(i) = rank + i; }
  );
  exec.fence();

  KokkosComm::Experimental::reduce(comm, sv, rv, root, KokkosComm::Sum{}).wait();

  if (rank == root) {
    int errs = 0;
    Kokkos::parallel_reduce(
        rv.extent(0),
        KOKKOS_LAMBDA(const int i, int& lsum) { lsum += (rv(i) != ((size * (size - 1)) / 2 + (size * i))); }, errs
    );
    EXPECT_EQ(errs, 0);
  }
}

#if defined(KOKKOSCOMM_ENABLE_MPI)
using MpiScalarTypes =
    testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
#endif
#if defined(KOKKOSCOMM_ENABLE_NCCL) || defined(KOKKOSCOMM_ENABLE_RCCL)
using XcclScalarTypes = testing::Types<float, double, int, int64_t>;
#endif

#define KC_DEFINE_REDUCE_TESTS(SuiteName, TypeList, CommSpace) \
  template <typename T>                                        \
  class SuiteName : public ReduceFixture<T> {};                \
                                                               \
  TYPED_TEST_SUITE(SuiteName, TypeList);                       \
                                                               \
  TYPED_TEST(SuiteName, Contiguous1D) { reduce_contig_1d<CommSpace, typename TestFixture::Scalar>(); }

#if defined(KOKKOSCOMM_ENABLE_MPI)
KC_DEFINE_REDUCE_TESTS(MpiReduce, MpiScalarTypes, KokkosComm::MpiSpace)
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL)
KC_DEFINE_REDUCE_TESTS(NcclReduce, XcclScalarTypes, KokkosComm::Experimental::NcclSpace)
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
KC_DEFINE_REDUCE_TESTS(RcclReduce, XcclScalarTypes, KokkosComm::Experimental::RcclSpace)
#endif

#undef KC_DEFINE_REDUCE_TESTS

}  // namespace
