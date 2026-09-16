// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/utils.hpp"
#endif

namespace {

using Ex = Kokkos::DefaultExecutionSpace;
using Co = KokkosComm::DefaultCommunicationSpace;

template <typename T>
class Broadcast : public testing::Test {
 public:
  using Scalar = T;
};
#if defined(KOKKOSCOMM_ENABLE_NCCL)
using ScalarTypes = testing::Types<float, double, int, int64_t>;
#else
using ScalarTypes =
    testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
#endif
TYPED_TEST_SUITE(Broadcast, ScalarTypes);

template <typename Scalar>
auto broadcast_0d() -> void {
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

template <typename Scalar>
auto broadcast_contig_1d() -> void {
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

TYPED_TEST(Broadcast, 0D) { broadcast_0d<typename TestFixture::Scalar>(); }
TYPED_TEST(Broadcast, Contiguous1D) { broadcast_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
