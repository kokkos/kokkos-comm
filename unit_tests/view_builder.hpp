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

#pragma once

#include <Kokkos_Core.hpp>

struct contig {};
struct noncontig {};

template <typename T, int RANK>
struct ViewBuilder;

template <typename T>
struct ViewBuilder<T, 1> {
  static auto view(noncontig, int e0) {
    // this is C-style layout, i.e. v(0,0) is next to v(0,1)
    Kokkos::View<T**, Kokkos::LayoutRight> v("", e0, 2);
    return Kokkos::subview(v, Kokkos::ALL, 1);  // take column 1
  }

  static auto view(contig, int e0) { return Kokkos::View<T*>("", e0); }
};

template <typename T>
struct ViewBuilder<T, 2> {
  static auto view(noncontig, int e0, int e1) {
    Kokkos::View<T***, Kokkos::LayoutRight> v("", e0, e1, 2);
    return Kokkos::subview(v, Kokkos::ALL, Kokkos::ALL, 1);
  }

  static auto view(contig, int e0, int e1) { return Kokkos::View<T**>("", e0, e1); }
};
