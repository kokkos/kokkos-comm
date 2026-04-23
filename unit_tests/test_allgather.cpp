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
class AllGather : public testing::Test {
 public:
  using Scalar = T;
};
#if defined(KOKKOSCOMM_ENABLE_NCCL)
using ScalarTypes = testing::Types<float, double, int, int64_t>;
#else
using ScalarTypes =
    testing::Types<float, double, Kokkos::complex<float>, Kokkos::complex<double>, int, unsigned, int64_t, size_t>;
#endif
TYPED_TEST_SUITE(AllGather, ScalarTypes);

template <typename Scalar>
auto allgather_0d() -> void {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto exec      = Ex();
  auto comm      = KokkosComm::Communicator<>::from_raw(raw_comm, exec);
  const int size = comm.size();
  const int rank = comm.rank();

  Kokkos::View<Scalar> sv("sv");
  Kokkos::View<Scalar *> rv("rv", size);

  // Prepare send view, 1 element per sender: their rank
  Kokkos::parallel_for(
      Kokkos::RangePolicy(exec, 0, sv.extent(0)), KOKKOS_LAMBDA(const int) { sv() = rank; }
  );
  exec.fence();

  KokkosComm::Experimental::allgather(comm, sv, rv).wait();

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(const int src, int &lsum) { lsum += rv(src) != src; }, errs
  );
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
auto allgather_contig_1d() -> void {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  auto raw_comm = nccl_ctx.comm();
#else
  auto raw_comm = MPI_COMM_WORLD;
#endif
  auto exec      = Ex();
  auto comm      = KokkosComm::Communicator<>::from_raw(raw_comm, exec);
  const int size = comm.size();
  const int rank = comm.rank();

  const int n_contrib = 100;
  Kokkos::View<Scalar *> sv("sv", n_contrib);
  Kokkos::View<Scalar *> rv("rv", size * n_contrib);

  // Prepare send view
  Kokkos::parallel_for(
      Kokkos::RangePolicy(exec, 0, sv.extent(0)), KOKKOS_LAMBDA(const int i) { sv(i) = rank + i; }
  );
  exec.fence();

  KokkosComm::Experimental::allgather(comm, sv, rv).wait();

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(const int i, int &lsum) {
        const int src = i / n_contrib;
        const int j   = i % n_contrib;
        lsum += rv(i) != src + j;
      },
      errs
  );
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(AllGather, 0D) { allgather_0d<typename TestFixture::Scalar>(); }
TYPED_TEST(AllGather, Contiguous1D) { allgather_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
