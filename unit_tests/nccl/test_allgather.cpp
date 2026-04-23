// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/KokkosComm.hpp>

#include "utils.hpp"

namespace {

template <typename T>
class AllGather : public testing::Test {
 public:
  using Scalar = T;
};
using ScalarTypes = ::testing::Types<int, int64_t, float, double>;

TYPED_TEST_SUITE(AllGather, ScalarTypes);

template <typename Scalar>
auto allgather_0d() -> void {
  auto nccl_ctx   = test_utils::nccl::Ctx::init();
  const auto exec = Kokkos::Cuda(nccl_ctx.stream());
  const auto comm = nccl_ctx.comm();
  const int size  = nccl_ctx.size();
  const int rank  = nccl_ctx.rank();
  const int root  = 0;

  Kokkos::View<Scalar> sv("sv");
  Kokkos::View<Scalar*> rv("rv", size);

  // Prepare send view, 1 element per sender: their rank
  Kokkos::parallel_for(
      Kokkos::RangePolicy(exec, 0, sv.extent(0)), KOKKOS_LAMBDA(const int) { sv() = rank; }
  );

  // Using the same execution space for both operations lets us not need an explicit `fence`
  KokkosComm::Experimental::nccl::allgather(exec, sv, rv, comm).wait();

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(const int src, int& lsum) { lsum += rv(src) != src; }, errs
  );
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
auto allgather_contig_1d() -> void {
  auto nccl_ctx   = test_utils::nccl::Ctx::init();
  const auto exec = Kokkos::Cuda(nccl_ctx.stream());
  const auto comm = nccl_ctx.comm();
  const int size  = nccl_ctx.size();
  const int rank  = nccl_ctx.rank();
  const int root  = 0;

  const int n_contrib = 100;
  Kokkos::View<Scalar*> sv("sv", n_contrib);
  Kokkos::View<Scalar*> rv("rv", size * n_contrib);

  // Prepare send view
  Kokkos::parallel_for(
      Kokkos::RangePolicy(exec, 0, sv.extent(0)), KOKKOS_LAMBDA(const int i) { sv(i) = rank + i; }
  );

  // Using the same execution space for both operations lets us not need an explicit `fence`
  KokkosComm::Experimental::nccl::allgather(exec, sv, rv, comm).wait();

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(const int i, int& lsum) {
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
