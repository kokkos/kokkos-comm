// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/KokkosComm.hpp>

#include "utils.hpp"

namespace {

using ExecSpace = Kokkos::Cuda;
using CommSpace = KokkosComm::Experimental::NcclSpace;

template <typename T>
class AllToAll : public testing::Test {
 public:
  using Scalar = T;
};
using ScalarTypes = ::testing::Types<int, int64_t, float, double>;

TYPED_TEST_SUITE(AllToAll, ScalarTypes);

template <typename Scalar>
auto alltoall_contig_1d() -> void {
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  ExecSpace space(nccl_ctx.stream());
  KokkosComm::Handle<ExecSpace, CommSpace> h(space, nccl_ctx.comm());
  int rank = h.rank();
  int size = h.size();

  int n_contrib = 100;
  Kokkos::View<Scalar *> sv("sv", size * n_contrib);
  Kokkos::View<Scalar *> rv("rv", size * n_contrib);

  // Prepare send view
  Kokkos::parallel_for(
      Kokkos::RangePolicy(space, 0, sv.extent(0)), KOKKOS_LAMBDA(const int i) { sv(i) = rank + i; });
  // Using the same execution space for both operations lets us not need an explicit `fence`
  KokkosComm::Experimental::alltoall(h, sv, rv, n_contrib);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(const int i, int &lsum) {
        const int src = i / n_contrib;                       // who sent this data
        const int j   = rank * n_contrib + (i % n_contrib);  // what index i was at the source
        lsum += rv(i) != src + j;
      },
      errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(AllToAll, Contiguous1D) { alltoall_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
