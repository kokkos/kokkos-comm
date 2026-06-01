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
class AllToAll : public testing::Test {
 public:
  using Scalar = T;
};
#if defined(KOKKOSCOMM_ENABLE_NCCL)
using ScalarTypes = testing::Types<float, double, int, int64_t>;
#else
using ScalarTypes =
    testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
#endif
TYPED_TEST_SUITE(AllToAll, ScalarTypes);

template <typename Scalar>
auto alltoall_contig_1d() -> void {
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

TYPED_TEST(AllToAll, Contiguous1D) { alltoall_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
