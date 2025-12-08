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
class Reduce : public testing::Test {
 public:
  using Scalar = T;
};
using ScalarTypes = ::testing::Types<int, int64_t, float, double>;

TYPED_TEST_SUITE(Reduce, ScalarTypes);

/// Each rank fills its sendbuf[i] with `rank + i`
/// operation is sum, so recvbuf[i] should be sum(0..size) + i * size
template <typename Scalar>
auto reduce_contig_1d() -> void {
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  ExecSpace space(nccl_ctx.stream());
  KokkosComm::Handle<ExecSpace, CommSpace> h(space, nccl_ctx.comm());
  int rank = h.rank();
  int size = h.size();
  int root = 0;

  int n_contrib = 100;
  Kokkos::View<Scalar *> sv("sv", n_contrib);
  Kokkos::View<Scalar *> rv("rv", 0);
  if (rank == root) {
    Kokkos::resize(rv, size);
  }

  // Prepare send buffer
  Kokkos::parallel_for(
      Kokkos::RangePolicy(space, 0, sv.extent(0)), KOKKOS_LAMBDA(const int i) { sv(i) = rank + i; });
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::Experimental::reduce(h, sv, rv, root, KokkosComm::Sum{});
  KokkosComm::wait(req);

  if (rank == root) {
    int errs;
    Kokkos::parallel_reduce(
        rv.extent(0),
        KOKKOS_LAMBDA(const int i, int &lsum) {
          Scalar acc = 0;
          for (int r = 0; r < size; ++r) {
            acc += r + i;
          }
          lsum += rv(i) != acc;
        },
        errs);
    ASSERT_EQ(errs, 0);
  }
}

TYPED_TEST(Reduce, Contiguous1D) { reduce_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
