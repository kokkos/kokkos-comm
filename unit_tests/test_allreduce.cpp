// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/utils.hpp"
#endif

namespace {

using Ex = Kokkos::DefaultExecutionSpace;
using Co = KokkosComm::DefaultCommunicationSpace;

template <typename T>
class AllReduce : public testing::Test {
 public:
  using Scalar = T;
};
#if defined(KOKKOSCOMM_ENABLE_NCCL)
using ScalarTypes = testing::Types<float, double, int, int64_t>;
#else
using ScalarTypes =
    testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
#endif
TYPED_TEST_SUITE(AllReduce, ScalarTypes);

template <typename Scalar>
auto allreduce_0d() -> void {
// FIXME_EXTERNAL #215
#if defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
  GTEST_SKIP() << "Unimplemented test for Open MPI + CUDA/HIP";
#else

#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto& nccl_ctx = test_utils::NcclCtx::get();
  auto raw_comm  = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto exec      = Ex();
  auto comm      = KokkosComm::Communicator<>::from_raw(raw_comm, exec);
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
#endif
}

template <typename Scalar>
auto allreduce_contig_1d() -> void {
// FIXME_EXTERNAL #215
#if defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
  GTEST_SKIP() << "Unimplemented test for Open MPI + CUDA/HIP";
#else

#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto& nccl_ctx = test_utils::NcclCtx::get();
  auto raw_comm  = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto exec      = Ex();
  auto comm      = KokkosComm::Communicator<>::from_raw(raw_comm, exec);
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
#endif
}

TYPED_TEST(AllReduce, 0D) { allreduce_0d<typename TestFixture::Scalar>(); }
TYPED_TEST(AllReduce, Contiguous1D) { allreduce_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
