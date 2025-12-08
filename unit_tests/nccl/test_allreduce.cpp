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
class AllReduce : public testing::Test {
 public:
  using Scalar = T;
};
using ScalarTypes = ::testing::Types<int, int64_t, float, double>;

TYPED_TEST_SUITE(AllReduce, ScalarTypes);

template <typename Scalar>
auto allreduce_0d() -> void {
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  ExecSpace space(nccl_ctx.stream());
  KokkosComm::Handle<ExecSpace, CommSpace> h(space, nccl_ctx.comm());
  int rank = h.rank();
  int size = h.size();

  Kokkos::View<Scalar> sv("sv");
  Kokkos::View<Scalar> rv("rv", size);

  // Prepare send buffer
  Kokkos::parallel_for(
      Kokkos::RangePolicy(space, 0, sv.extent(0)), KOKKOS_LAMBDA(const int) { sv() = rank; });
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::Experimental::allreduce(h, sv, rv, KokkosComm::Sum{});
  KokkosComm::wait(req);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(const int, int &lsum) { lsum += (rv() != size * (size - 1) / 2); }, errs);
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
auto allreduce_contig_1d() -> void {
  auto nccl_ctx = test_utils::nccl::Ctx::init();
  ExecSpace space(nccl_ctx.stream());
  KokkosComm::Handle<ExecSpace, CommSpace> h(space, nccl_ctx.comm());
  int rank = h.rank();
  int size = h.size();

  int n_contrib = 10;
  Kokkos::View<Scalar *> sv("sv", n_contrib);
  Kokkos::View<Scalar *> rv("rv", size);

  // Prepare send buffer
  Kokkos::parallel_for(
      Kokkos::RangePolicy(space, 0, sv.extent(0)), KOKKOS_LAMBDA(const int i) { sv(i) = rank + i; });
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::Experimental::allreduce(h, sv, rv, KokkosComm::Sum{});
  KokkosComm::wait(req);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(const int i, int &lsum) { lsum += (rv(i) != size * (size - 1) / 2 + size * i); },
      errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(AllReduce, 0D) { allreduce_0d<typename TestFixture::Scalar>(); }
TYPED_TEST(AllReduce, Contiguous1D) { allreduce_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
