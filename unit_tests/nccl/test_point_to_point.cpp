// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstdint>

#include <gtest/gtest.h>
#include <nccl.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/KokkosComm.hpp>

#include "utils.hpp"

namespace {

using ExecSpace = Kokkos::Cuda;
using CommSpace = KokkosComm::Experimental::NcclSpace;

template <typename T>
class PointToPoint : public testing::Test {
 public:
  using Scalar = T;
};
using ScalarTypes = ::testing::Types<int, int64_t, float, double>;

TYPED_TEST_SUITE(PointToPoint, ScalarTypes);

template <typename Scalar>
auto p2p_contig_1d() -> void {
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  ExecSpace space(nccl_ctx.stream());
  KokkosComm::Handle<ExecSpace, CommSpace> h(space, nccl_ctx.comm());
  int rank = h.rank();
  int size = h.size();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  int src = 0;
  int dst = 1;

  Kokkos::View<Scalar *> v("v", 10'000);
  if (rank == src) {
    // Prepare send view
    Kokkos::parallel_for(
        Kokkos::RangePolicy(space, 0, v.extent(0)), KOKKOS_LAMBDA(const int i) { v(i) = i; });
    // Using the same execution space for both operations lets us not need an explicit `fence`
    auto req = KokkosComm::send(h, v, dst);
    KokkosComm::wait(req);
  } else if (rank == dst) {
    auto req = KokkosComm::recv(h, v, src);
    KokkosComm::wait(req);

    int errs;
    Kokkos::parallel_reduce(
        v.extent(0), KOKKOS_LAMBDA(const int i, int &lsum) { lsum += v(i) != Scalar(i); }, errs);
    ASSERT_EQ(errs, 0);
  }
}

template <typename Scalar>
auto p2p_noncontig_1d() -> void {
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  ExecSpace space(nccl_ctx.stream());
  KokkosComm::Handle<ExecSpace, CommSpace> h(space, nccl_ctx.comm());
  int rank = h.rank();
  int size = h.size();
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }
  int src = 0;
  int dst = 1;

  Kokkos::View<Scalar **, Kokkos::LayoutRight> v("v", 100, 100);
  auto sv = Kokkos::subview(v, Kokkos::ALL, 2);  // take column 2 (non-contiguous)
  if (rank == src) {
    // Prepare send view
    Kokkos::parallel_for(
        Kokkos::RangePolicy(space, 0, sv.extent(0)), KOKKOS_LAMBDA(const int i) { sv(i) = i; });
    // Using the same execution space for both operations lets us not need an explicit `fence`
    auto req = KokkosComm::send(h, sv, dst);
    KokkosComm::wait(req);
  } else if (rank == dst) {
    auto req = KokkosComm::recv(h, sv, src);
    KokkosComm::wait(req);

    int errs;
    Kokkos::parallel_reduce(
        sv.extent(0), KOKKOS_LAMBDA(const int i, int &lsum) { lsum += sv(i) != Scalar(i); }, errs);
    ASSERT_EQ(errs, 0);
  }
}

TYPED_TEST(PointToPoint, Contiguous1D) { p2p_contig_1d<typename TestFixture::Scalar>(); }
TYPED_TEST(PointToPoint, NonContiguous1D) { p2p_noncontig_1d<typename TestFixture::Scalar>(); }

}  // namespace
