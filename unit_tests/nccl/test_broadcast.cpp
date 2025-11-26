// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <KokkosComm/KokkosComm.hpp>

#include "utils.hpp"

namespace {

using ExecSpace = Kokkos::Cuda;
using CommSpace = KokkosComm::Experimental::NcclSpace;

template <typename T>
class Broadcast : public testing::Test {
 public:
  using Scalar = T;
};
using ScalarTypes = ::testing::Types<int, int64_t, float, double>;

TYPED_TEST_SUITE(Broadcast, ScalarTypes);

template <typename Scalar>
auto broadcast_0d() -> void {
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  ExecSpace space(nccl_ctx.stream());
  KokkosComm::Handle<ExecSpace, CommSpace> h(space, nccl_ctx.comm());
  int rank = h.rank();
  int size = h.size();
  int root = 0;

  Kokkos::View<Scalar> v("v");
  if (rank == root) {
    // Prepare broadcast view
    Kokkos::parallel_for(
        Kokkos::RangePolicy(space, 0, v.extent(0)), KOKKOS_LAMBDA(const int) { v() = size; });
  }
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::Experimental::broadcast(h, v, root);
  KokkosComm::wait(req);

  int errs;
  Kokkos::parallel_reduce(
      v.extent(0), KOKKOS_LAMBDA(const int, int& lsum) { lsum += v() != size; }, errs);
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
auto broadcast_contig_1d() -> void {
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  ExecSpace space(nccl_ctx.stream());
  KokkosComm::Handle<ExecSpace, CommSpace> h(space, nccl_ctx.comm());
  int rank = h.rank();
  int size = h.size();
  int root = 0;

  Kokkos::View<Scalar*> v("v", 100);
  if (rank == root) {
    // Prepare broadcast view
    Kokkos::parallel_for(
        Kokkos::RangePolicy(space, 0, v.extent(0)), KOKKOS_LAMBDA(const int i) { v(i) = size + i; });
  }
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::Experimental::broadcast(h, v, root);
  KokkosComm::wait(req);

  int errs;
  Kokkos::parallel_reduce(
      v.extent(0), KOKKOS_LAMBDA(const int i, int& lsum) { lsum += (v(i) != size + i); }, errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(Broadcast, 0D) { broadcast_0d<typename TestFixture::Scalar>(); }
TYPED_TEST(Broadcast, Contiguous1D) { broadcast_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
