//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2025) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <gtest/gtest.h>
#include <type_traits>
#include <KokkosComm/KokkosComm.hpp>

namespace {

using namespace KokkosComm::mpi;

template <typename T>
class WindowTest : public testing::Test {
 public:
  using Scalar = T;
};

// Kokkos::complex<float> excluded: MPI_Accumulate + MPI_SUM fails with MPI_COMPLEX (Fortran type) in Open MPI
// using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<double>>;
TYPED_TEST_SUITE(WindowTest, ScalarTypes);

template <typename Scalar>
void test_fence_put() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  window.fence();

  if (rank == 0) {
    Scalar value = static_cast<Scalar>(99);
    window.put(&value, 1, 1, 0);
  }

  window.fence();

  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(99));
  }
}

TYPED_TEST(WindowTest, FencePut) { test_fence_put<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_fence_get() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);
  Scalar value = static_cast<Scalar>(-1);

  window.fence();

  if (rank == 1) {
    window.get(&value, 1, 0, 0);
  }

  window.fence();

  if (rank == 1) {
    EXPECT_EQ(value, static_cast<Scalar>(0));
  }
}

TYPED_TEST(WindowTest, FenceGet) { test_fence_get<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_fence_accumulate() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  window.fence();

  if (rank == 0) {
    Scalar value = static_cast<Scalar>(10);
    window.accumulate(&value, 1, 1, 0, MPI_SUM);
  }

  window.fence();

  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(11));
  }
}

TYPED_TEST(WindowTest, FenceAccumulate) { test_fence_accumulate<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_lock_unlock_exclusive_put() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  if (rank == 0) {
    window.lock(KokkosComm::Window<decltype(data)>::LockType::Exclusive, 1);
    Scalar value = static_cast<Scalar>(0);
    window.put(&value, 1, 1, 0);
    window.unlock(1);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(0));
  }
}

TYPED_TEST(WindowTest, LockUnlockExclusivePut) { test_lock_unlock_exclusive_put<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_lock_unlock_exclusive_get() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank * 100);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  if (rank == 1) {
    window.lock(KokkosComm::Window<decltype(data)>::LockType::Exclusive, 0);
    Scalar value = static_cast<Scalar>(-1);
    window.get(&value, 1, 0, 0);
    window.unlock(0);
    EXPECT_EQ(value, static_cast<Scalar>(0));
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    EXPECT_EQ(data(0), static_cast<Scalar>(0));
  }
}

TYPED_TEST(WindowTest, LockUnlockExclusiveGet) { test_lock_unlock_exclusive_get<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_lock_unlock_exclusive_accumulate() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  if (rank == 0) {
    window.lock(KokkosComm::Window<decltype(data)>::LockType::Exclusive, 1);
    Scalar value = static_cast<Scalar>(10);
    window.accumulate(&value, 1, 1, 0, MPI_SUM);
    window.unlock(1);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(11));
  }
}

TYPED_TEST(WindowTest, LockUnlockExclusiveAccumulate) {
  test_lock_unlock_exclusive_accumulate<typename TestFixture::Scalar>();
}

template <typename Scalar>
void test_lock_unlock_shared_put() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", size + 1);
  for (int i = 0; i < size + 1; ++i) {
    data(i) = static_cast<Scalar>(-1);
  }

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  window.lock(KokkosComm::Window<decltype(data)>::LockType::Shared, 0);
  Scalar value = static_cast<Scalar>((rank + 1) * 100);
  window.put(&value, 1, 0, rank);
  window.unlock(0);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    for (int r = 0; r < size; ++r) {
      EXPECT_EQ(data(r), static_cast<Scalar>((r + 1) * 100));
    }
    EXPECT_EQ(data(size), static_cast<Scalar>(-1));
  }
}

TYPED_TEST(WindowTest, LockUnlockSharedPut) { test_lock_unlock_shared_put<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_lock_unlock_shared_get() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>((rank + 1) * 100);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  window.lock(KokkosComm::Window<decltype(data)>::LockType::Shared, 0);
  Scalar value = static_cast<Scalar>(-1);
  window.get(&value, 1, 0, 0);
  window.unlock(0);
  EXPECT_EQ(value, static_cast<Scalar>(100));

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    EXPECT_EQ(data(0), static_cast<Scalar>(100));
  }
}

TYPED_TEST(WindowTest, LockUnlockSharedGet) { test_lock_unlock_shared_get<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_lock_unlock_shared_accumulate() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(0);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  window.lock(KokkosComm::Window<decltype(data)>::LockType::Shared, 0);
  Scalar value = static_cast<Scalar>(10);
  window.accumulate(&value, 1, 0, 0, MPI_SUM);
  window.unlock(0);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    EXPECT_EQ(data(0), static_cast<Scalar>(10 * size));
  }
}

TYPED_TEST(WindowTest, LockUnlockSharedAccumulate) {
  test_lock_unlock_shared_accumulate<typename TestFixture::Scalar>();
}

template <typename Scalar>
void test_lock_all_unlock_all_put() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 2);
  data(0) = static_cast<Scalar>(-1);
  data(1) = static_cast<Scalar>(-1);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  if (rank == 0) {
    window.lock_all();
    Scalar value1 = static_cast<Scalar>(100);
    Scalar value2 = static_cast<Scalar>(200);
    window.put(&value1, 1, 1, 0);
    window.put(&value2, 1, 0, 1);
    window.unlock_all();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    EXPECT_EQ(data(0), static_cast<Scalar>(-1));
    EXPECT_EQ(data(1), static_cast<Scalar>(200));
  }
  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(100));
    EXPECT_EQ(data(1), static_cast<Scalar>(-1));
  }
}

TYPED_TEST(WindowTest, LockAllUnlockAllPut) { test_lock_all_unlock_all_put<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_flush_put() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  Scalar value = static_cast<Scalar>(100);

  if (rank == 0) {
    window.lock(KokkosComm::Window<decltype(data)>::LockType::Shared, 1);
    window.put(&value, 1, 1, 0);
    window.flush(1);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(100));
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    window.unlock(1);
  }
}

TYPED_TEST(WindowTest, FlushPut) { test_flush_put<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_flush_local_put() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  if (rank == 0) {
    window.lock(KokkosComm::Window<decltype(data)>::LockType::Shared, 1);
    Scalar value = static_cast<Scalar>(100);
    window.put(&value, 1, 1, 0);
    window.flush_local(1);
    value = static_cast<Scalar>(999);
    window.unlock(1);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(100));
  }
}

TYPED_TEST(WindowTest, FlushLocalPut) { test_flush_local_put<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_flush_all_put() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 2);
  data(0) = static_cast<Scalar>(-1);
  data(1) = static_cast<Scalar>(-1);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  if (rank == 0) {
    window.lock_all();
    Scalar value1 = static_cast<Scalar>(100);
    Scalar value2 = static_cast<Scalar>(200);
    window.put(&value1, 1, 1, 0);
    window.put(&value2, 1, 0, 1);
    window.flush_all();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    EXPECT_EQ(data(0), static_cast<Scalar>(-1));
    EXPECT_EQ(data(1), static_cast<Scalar>(200));
  }
  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(100));
    EXPECT_EQ(data(1), static_cast<Scalar>(-1));
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    window.unlock_all();
  }
}

TYPED_TEST(WindowTest, FlushAllPut) { test_flush_all_put<typename TestFixture::Scalar>(); }

template <typename Scalar>
void test_flush_local_all_put() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 2);
  data(0) = static_cast<Scalar>(rank);
  data(1) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  if (rank == 0) {
    window.lock_all();
    Scalar value1 = static_cast<Scalar>(100);
    Scalar value2 = static_cast<Scalar>(200);
    window.put(&value1, 1, 1, 0);
    window.put(&value2, 1, 0, 1);
    window.flush_local_all();
    value1 = static_cast<Scalar>(999);
    value2 = static_cast<Scalar>(999);
    window.unlock_all();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 1) {
    EXPECT_EQ(data(0), static_cast<Scalar>(100));
  }
  if (rank == 0) {
    EXPECT_EQ(data(1), static_cast<Scalar>(200));
  }
}

TYPED_TEST(WindowTest, FlushLocalAllPut) { test_flush_local_all_put<typename TestFixture::Scalar>(); }

template <typename Scalar>
void pscw_put() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }
  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  MPI_Group world_group, origin_group, target_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  int origin_ranks[1] = {1};
  int target_ranks[1] = {0};

  MPI_Group_incl(world_group, 1, origin_ranks, &origin_group);
  MPI_Group_incl(world_group, 1, target_ranks, &target_group);

  if (rank == 0) {
    window.post(origin_group);
  }

  if (rank == 1) {
    window.start(target_group);
    Scalar value = static_cast<Scalar>(99);
    window.put(&value, 1, 0, 0);
    window.complete();
  }

  if (rank == 0) {
    window.wait();
    EXPECT_EQ(data(0), static_cast<Scalar>(99));
  }

  MPI_Group_free(&origin_group);
  MPI_Group_free(&target_group);
  MPI_Group_free(&world_group);
}

TYPED_TEST(WindowTest, PSCWPut) { pscw_put<typename TestFixture::Scalar>(); }

template <typename Scalar>
void pscw_get() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }
  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  MPI_Group world_group, origin_group, target_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  int origin_ranks[1] = {1};
  int target_ranks[1] = {0};

  MPI_Group_incl(world_group, 1, origin_ranks, &origin_group);
  MPI_Group_incl(world_group, 1, target_ranks, &target_group);

  if (rank == 0) {
    window.post(origin_group);
  }

  if (rank == 1) {
    window.start(target_group);
    Scalar value = static_cast<Scalar>(-1);
    window.get(&value, 1, 0, 0);
    window.complete();
    EXPECT_EQ(value, static_cast<Scalar>(0));
  }

  if (rank == 0) {
    window.wait();
    EXPECT_EQ(data(0), static_cast<Scalar>(0));
  }

  MPI_Group_free(&origin_group);
  MPI_Group_free(&target_group);
  MPI_Group_free(&world_group);
}

TYPED_TEST(WindowTest, PSCWGet) { pscw_get<typename TestFixture::Scalar>(); }

template <typename Scalar>
void pscw_accumulate() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  Kokkos::View<Scalar*, Kokkos::HostSpace> data("data", 1);
  data(0) = static_cast<Scalar>(rank);

  KokkosComm::Window<decltype(data)> window(data, MPI_COMM_WORLD);

  MPI_Group world_group, origin_group, target_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  int origin_ranks[1] = {0};
  int target_ranks[1] = {1};

  MPI_Group_incl(world_group, 1, origin_ranks, &origin_group);
  MPI_Group_incl(world_group, 1, target_ranks, &target_group);

  if (rank == 1) {
    window.post(origin_group);
  }

  if (rank == 0) {
    window.start(target_group);
    Scalar value = static_cast<Scalar>(10);
    window.accumulate(&value, 1, 1, 0, MPI_SUM);
    window.complete();
  }

  if (rank == 1) {
    window.wait();
    EXPECT_EQ(data(0), static_cast<Scalar>(11));
  }

  MPI_Group_free(&origin_group);
  MPI_Group_free(&target_group);
  MPI_Group_free(&world_group);
}

TYPED_TEST(WindowTest, PSCWAccumulate) { pscw_accumulate<typename TestFixture::Scalar>(); }

}  // namespace
