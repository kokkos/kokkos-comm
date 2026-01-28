// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <type_traits>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/impl/host_staging.hpp>

namespace {

template <KokkosComm::KokkosExecutionSpace E>
auto test_stage_for_returns_host_accessible_view() -> void {
  auto space = E{};

  constexpr int N = 100;
  Kokkos::View<int*, typename E::memory_space> view("v", N);
  Kokkos::parallel_for(
      "fill", Kokkos::RangePolicy(space, 0, N), KOKKOS_LAMBDA(int i) { view(i) = i; });
  space.fence();

  auto staged = KokkosComm::Impl::stage_for(view);
  space.fence();

  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename decltype(staged)::memory_space>::accessible);

  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(staged(i), i);
  }
}

TEST(StagingComptimeTest, NeedsStagingTraitIsCorrect) {
  using HostView = Kokkos::View<int*, Kokkos::HostSpace>;
  EXPECT_FALSE(KokkosComm::Impl::needs_staging_v<HostView>);

#ifdef KOKKOS_ENABLE_CUDA
  using CudaView = Kokkos::View<int*, Kokkos::CudaSpace>;
  EXPECT_TRUE(KokkosComm::Impl::needs_staging_v<CudaView>);
  using UVMView = Kokkos::View<int*, Kokkos::CudaUVMSpace>;
  EXPECT_FALSE(KokkosComm::Impl::needs_staging_v<UVMView>);
#endif

#ifdef KOKKOS_ENABLE_HIP
  using HIPView = Kokkos::View<int*, Kokkos::HIPSpace>;
  EXPECT_TRUE(KokkosComm::Impl::needs_staging_v<HIPView>);
  using HIPManagedView = Kokkos::View<int*, Kokkos::HIPManagedSpace>;
  EXPECT_FALSE(KokkosComm::Impl::needs_staging_v<HIPManagedView>);
#endif
}

TEST(StagingTest, StageForPreservesDataPointerForHostViews) {
  auto space = Kokkos::DefaultHostExecutionSpace{};

  constexpr int N = 100;
  Kokkos::View<int*, typename decltype(space)::memory_space> host_view("v", N);

  auto staged = KokkosComm::Impl::stage_for(host_view);
  space.fence();
  EXPECT_EQ(staged.data(), host_view.data());
}

TEST(StagingTest, StageForCreatesIndependentCopyForDeviceViews) {
  if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>) {
    GTEST_SKIP() << "Default execution space is on host";
  } else {
    auto space = Kokkos::DefaultExecutionSpace{};

    constexpr int N = 100;
    Kokkos::View<int*, typename decltype(space)::memory_space> device_view("v", N);

    auto staged = KokkosComm::Impl::stage_for(device_view);
    space.fence();
    EXPECT_NE(reinterpret_cast<void*>(staged.data()), reinterpret_cast<void*>(device_view.data()));
  }
}

TEST(StagingTest, CopyBackTransfersData) {
  auto space = Kokkos::DefaultExecutionSpace{};

  constexpr int N = 100;
  Kokkos::View<int*, typename decltype(space)::memory_space> device_view("v", N);

  space.fence();
  auto staged = KokkosComm::Impl::stage_for(device_view);
  Kokkos::DefaultHostExecutionSpace().fence();
  for (int i = 0; i < N; ++i) {
    staged(i) = 2 * i;
  }

  KokkosComm::Impl::copy_back(space, device_view, staged);
  space.fence();

  auto check = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_view);
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(check(i), 2 * i);
  }
}

template <typename T>
class StagingTypedTest : public ::testing::Test {
 public:
  using Space = T;
};

using SpaceTypes = ::testing::Types<Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace>;
TYPED_TEST_SUITE(StagingTypedTest, SpaceTypes);

TYPED_TEST(StagingTypedTest, StageForReturnsHostAccessibleView) {
  test_stage_for_returns_host_accessible_view<typename TestFixture::Space>();
}

}  // namespace
