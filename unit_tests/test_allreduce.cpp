// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#include "communication_space_utils.hpp"

namespace {

template <typename T>
class AllReduceFixture : public testing::Test {
 public:
  using Scalar = T;
};

template <KokkosComm::CommunicationSpace CommSpace, typename Scalar>
auto allreduce_0d() -> void {
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

  Kokkos::View<Scalar> sv("sv");
  Kokkos::View<Scalar> rv("rv");

  // Prepare send buffer
  Kokkos::parallel_for(
      Kokkos::RangePolicy(exec, 0, sv.extent(0)), KOKKOS_LAMBDA(const int) { sv() = rank; }
  );
  exec.fence();

  KokkosComm::Experimental::allreduce(comm, sv, rv, KokkosComm::Sum{}).wait();

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(const int, int& lsum) { lsum += (rv() != size * (size - 1) / 2); }, errs
  );
  EXPECT_EQ(errs, 0);
}

template <KokkosComm::CommunicationSpace CommSpace, typename Scalar>
auto allreduce_contig_1d() -> void {
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

  const int n_contrib = 10;
  Kokkos::View<Scalar*> sv("sv", n_contrib);
  Kokkos::View<Scalar*> rv("rv", n_contrib);

  // Prepare send buffer
  Kokkos::parallel_for(
      Kokkos::RangePolicy(exec, 0, sv.extent(0)), KOKKOS_LAMBDA(const int i) { sv(i) = rank + i; }
  );
  exec.fence();

  KokkosComm::Experimental::allreduce(comm, sv, rv, KokkosComm::Sum{}).wait();

  int errs;
  // Fill in this reduction which verifies that each element computed the correct value
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(const int i, int& lsum) { lsum += (rv(i) != ((size * (size - 1)) / 2 + (size * i))); }, errs
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

#define KC_DEFINE_ALLREDUCE_TESTS(SuiteName, TypeList, CommSpace)                        \
  template <typename T>                                                                  \
  class SuiteName : public AllReduceFixture<T> {};                                       \
                                                                                         \
  TYPED_TEST_SUITE(SuiteName, TypeList);                                                 \
                                                                                         \
  TYPED_TEST(SuiteName, 0D) { allreduce_0d<CommSpace, typename TestFixture::Scalar>(); } \
  TYPED_TEST(SuiteName, Contiguous1D) { allreduce_contig_1d<CommSpace, typename TestFixture::Scalar>(); }

#if defined(KOKKOSCOMM_ENABLE_MPI)
KC_DEFINE_ALLREDUCE_TESTS(MpiAllReduce, MpiScalarTypes, KokkosComm::MpiSpace)
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL)
KC_DEFINE_ALLREDUCE_TESTS(NcclAllReduce, XcclScalarTypes, KokkosComm::Experimental::NcclSpace)
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
KC_DEFINE_ALLREDUCE_TESTS(RcclAllReduce, XcclScalarTypes, KokkosComm::Experimental::RcclSpace)
#endif

#undef KC_DEFINE_ALLREDUCE_TESTS

}  // namespace
