//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
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
#include <algorithm>  // iota
#include <random>

#include "KokkosComm/KokkosComm.hpp"

namespace {

using namespace KokkosComm::mpi;

template <typename T>
class MpiWaitAll : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, double, Kokkos::complex<float>>;
TYPED_TEST_SUITE(MpiWaitAll, ScalarTypes);

template <KokkosComm::KokkosExecutionSpace ExecSpace, typename Scalar>
void wait_all() {
  using TestView = Kokkos::View<Scalar *>;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }

  constexpr size_t numMsg = 128;
  ExecSpace space;
  std::vector<KokkosComm::Req<>> reqs;
  std::vector<TestView> views;

  for (size_t i = 0; i < numMsg; ++i) {
    views.push_back(TestView(std::to_string(i), i));
  }

  constexpr unsigned int SEED = 31337;
  std::random_device rd;
  std::mt19937 g(SEED);

  // random send/recv order
  std::vector<size_t> order(numMsg);
  std::iota(order.begin(), order.end(), size_t(0));
  std::shuffle(order.begin(), order.end(), g);

  KokkosComm::Handle<ExecSpace, KokkosComm::Mpi> h(space, MPI_COMM_WORLD);

  if (0 == rank) {
    constexpr int dst = 1;

    for (size_t i : order) {
      reqs.push_back(KokkosComm::send(h, views[i], dst));
    }

    KokkosComm::wait_all(reqs);

  } else if (1 == rank) {
    constexpr int src = 0;

    for (size_t i : order) {
      reqs.push_back(KokkosComm::recv(h, views[i], src));
    }

    KokkosComm::wait_all(reqs);
  }
}

// TODO: test call on no requests

TYPED_TEST(MpiWaitAll, default_execution_space) {
  wait_all<Kokkos::DefaultExecutionSpace, typename TestFixture::Scalar>();
}

TYPED_TEST(MpiWaitAll, default_host_execution_space) {
  if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace>) {
    GTEST_SKIP() << "Skipping test: DefaultHostExecSpace = DefaultExecSpace";
  } else {
    wait_all<Kokkos::DefaultHostExecutionSpace, typename TestFixture::Scalar>();
  }
}

}  // namespace
