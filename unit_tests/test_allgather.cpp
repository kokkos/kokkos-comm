// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/utils.hpp"
#endif

namespace {

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
  using ExecSpace = Kokkos::Cuda;
  auto nccl_ctx   = test_utils::nccl::Ctx::init();
  KokkosComm::Handle<ExecSpace, KokkosComm::Experimental::NcclSpace> h(ExecSpace(), nccl_ctx.comm());
#else
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  KokkosComm::Handle<ExecSpace, KokkosComm::MpiSpace> h{};
#endif
  int rank = h.rank();
  int size = h.size();

  Kokkos::View<Scalar> sv("sv");
  Kokkos::View<Scalar *> rv("rv", size);

  // Prepare send view, 1 element per sender: their rank
  Kokkos::parallel_for(
      Kokkos::RangePolicy(ExecSpace(), 0, sv.extent(0)), KOKKOS_LAMBDA(const int) { sv() = rank; });
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::Experimental::allgather(h, sv, rv);
  KokkosComm::wait(req);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0), KOKKOS_LAMBDA(const int src, int &lsum) { lsum += rv(src) != src; }, errs);
  EXPECT_EQ(errs, 0);
}

template <typename Scalar>
auto allgather_contig_1d() -> void {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  using ExecSpace = Kokkos::Cuda;
  auto nccl_ctx   = test_utils::nccl::Ctx::init();
  KokkosComm::Handle<ExecSpace, KokkosComm::Experimental::NcclSpace> h(ExecSpace(), nccl_ctx.comm());
#else
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  KokkosComm::Handle<ExecSpace, KokkosComm::MpiSpace> h{};
#endif
  int rank = h.rank();
  int size = h.size();

  const int n_contrib = 100;
  Kokkos::View<Scalar *> sv("sv", n_contrib);
  Kokkos::View<Scalar *> rv("rv", size * n_contrib);

  // Prepare send view
  Kokkos::parallel_for(
      Kokkos::RangePolicy(ExecSpace(), 0, sv.extent(0)), KOKKOS_LAMBDA(const int i) { sv(i) = rank + i; });
  // Using the same execution space for both operations lets us not need an explicit `fence`
  auto req = KokkosComm::Experimental::allgather(h, sv, rv);
  KokkosComm::wait(req);

  int errs;
  Kokkos::parallel_reduce(
      rv.extent(0),
      KOKKOS_LAMBDA(const int i, int &lsum) {
        const int src = i / n_contrib;
        const int j   = i % n_contrib;
        lsum += rv(i) != src + j;
      },
      errs);
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(AllGather, 0D) { allgather_0d<typename TestFixture::Scalar>(); }
TYPED_TEST(AllGather, Contiguous1D) { allgather_contig_1d<typename TestFixture::Scalar>(); }

}  // namespace
