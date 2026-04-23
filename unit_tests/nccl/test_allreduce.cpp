// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/KokkosComm.hpp>

#include "utils.hpp"

namespace {

template <typename T>
class AllReduce : public testing::Test {
 public:
  using Scalar = T;
};
using ScalarTypes = ::testing::Types<int, int64_t, float, double>;

TYPED_TEST_SUITE(AllReduce, ScalarTypes);

template <typename Scalar>
auto allreduce_0d() -> void {
  auto nccl_ctx   = test_utils::nccl::Ctx::init();
  const auto exec = Kokkos::Cuda(nccl_ctx.stream());
  const auto comm = nccl_ctx.comm();
  const int size  = nccl_ctx.size();
  const int rank  = nccl_ctx.rank();
  const int root  = 0;

  Kokkos::View<Scalar> sv("sv");
  Kokkos::View<Scalar> rv("rv");

  // Prepare send buffer
  Kokkos::parallel_for(
      Kokkos::RangePolicy(exec, 0, sv.extent(0)), KOKKOS_LAMBDA(const int) { sv() = rank; }
  );

  // Using the same execution space for both operations lets us not need an explicit `fence`
  KokkosComm::Experimental::nccl::allreduce(exec, sv, rv, ncclSum, comm).wait();

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(const int, int &lsum) { lsum += (rv() != size * (size - 1) / 2); }, errs
  );
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
auto allreduce_contig_1d() -> void {
  auto nccl_ctx   = test_utils::nccl::Ctx::init();
  const auto exec = Kokkos::Cuda(nccl_ctx.stream());
  const auto comm = nccl_ctx.comm();
  const int size  = nccl_ctx.size();
  const int rank  = nccl_ctx.rank();
  const int root  = 0;

  const int n_contrib = 10;
  Kokkos::View<Scalar *> sv("sv", n_contrib);
  Kokkos::View<Scalar *> rv("rv", n_contrib);

  // Prepare send buffer
  Kokkos::parallel_for(
      Kokkos::RangePolicy(exec, 0, sv.extent(0)), KOKKOS_LAMBDA(const int i) { sv(i) = rank + i; }
  );

  // Using the same execution space for both operations lets us not need an explicit `fence`
  KokkosComm::Experimental::nccl::allreduce(exec, sv, rv, ncclSum, comm).wait();

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(const int i, int &lsum) { lsum += (rv(i) != size * (size - 1) / 2 + size * i); }, errs
  );
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(AllReduce, 0D) { allreduce_0d<typename TestFixture::Scalar>(); }
TYPED_TEST(AllReduce, Contiguous1D) { allreduce_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
