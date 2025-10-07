// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>

struct contig {};
struct noncontig {};

template <typename T, int RANK>
struct ViewBuilder;

template <typename T>
struct ViewBuilder<T, 1> {
  static auto view(noncontig, const std::string &name, int e0) {
    // this is C-style layout, i.e. v(0,0) is next to v(0,1)
    Kokkos::View<T **, Kokkos::LayoutRight> v(name, e0, 2);
    return Kokkos::subview(v, Kokkos::ALL, 1);  // take column 1
  }

  static auto view(contig, const std::string &name, int e0) { return Kokkos::View<T *>(name, e0); }
};

template <typename T>
struct ViewBuilder<T, 2> {
  static auto view(noncontig, const std::string &name, int e0, int e1) {
    Kokkos::View<T ***, Kokkos::LayoutRight> v(name, e0, e1, 2);
    return Kokkos::subview(v, Kokkos::ALL, Kokkos::ALL, 1);
  }

  static auto view(contig, const std::string &name, int e0, int e1) { return Kokkos::View<T **>(name, e0, e1); }
};
