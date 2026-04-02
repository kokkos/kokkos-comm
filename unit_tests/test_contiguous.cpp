// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <KokkosComm/impl/contiguous.hpp>
#include "KokkosComm/concepts.hpp"
#include "KokkosComm/traits.hpp"
#include "view_utils.hpp"

namespace {

template <KokkosComm::KokkosView View>
void test_contiguous_allocate(const View& non_contig) {
  // If `non_contig` isn't rank-0 (i.e., a scalar), assert that it is indeed non-contiguous
  if constexpr (View::rank >= 1) {
    EXPECT_FALSE(KokkosComm::is_contiguous(non_contig));
  }

  auto exec   = Kokkos::DefaultExecutionSpace{};
  auto contig = KokkosComm::Impl::allocate_contiguous_for(exec, "contig", non_contig);
  exec.fence();

  // Rank should be preserved
  static_assert(decltype(contig)::rank == View::rank, "");

  // Memory space should be preserved
  static_assert(std::is_same_v<typename decltype(contig)::memory_space, typename View::memory_space>, "");

  // Size should be preserved
  EXPECT_EQ(contig.size(), non_contig.size());

  // Allocation should be contiguous
  EXPECT_TRUE(KokkosComm::is_contiguous(contig));
}

template <size_t R>
using ct_usize = std::integral_constant<size_t, R>;

template <typename R>
class Contiguous : public ::testing::Test {
 public:
  static constexpr size_t rank = R::value;
};
// Only test up to rank-7 so we can build a non-contiguous View (Kokkos limits View rank to 8)
using Ranks = ::testing::
    Types<ct_usize<0>, ct_usize<1>, ct_usize<2>, ct_usize<3>, ct_usize<4>, ct_usize<5>, ct_usize<6>, ct_usize<7>>;

TYPED_TEST_SUITE(Contiguous, Ranks);

TYPED_TEST(Contiguous, Allocate) {
  constexpr size_t R = TestFixture::rank;
  constexpr size_t N = 10;
  auto non_contig    = [ R, N ]<size_t... Is>(std::index_sequence<Is...>) {
       return test_utils::build_view<double, R>(
        test_utils::NonContig{}, "non_contig", ((void)Is, std::integral_constant<size_t, N>{})...
    );
  }
  (std::make_index_sequence<R>{});

  test_contiguous_allocate(non_contig);
}

}  // namespace
