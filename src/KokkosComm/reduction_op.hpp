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

#include <KokkosComm/concepts.hpp>

namespace KokkosComm {

struct ReduceMaximum {};
template <>
struct KokkosComm::Impl::is_reduction_operator<ReduceMaximum> : public std::true_type {};

struct ReduceMinimum {};
template <>
struct KokkosComm::Impl::is_reduction_operator<ReduceMinimum> : public std::true_type {};

struct ReduceSum {};
template <>
struct KokkosComm::Impl::is_reduction_operator<ReduceSum> : public std::true_type {};

struct ReduceProduct {};
template <>
struct KokkosComm::Impl::is_reduction_operator<ReduceProduct> : public std::true_type {};

struct ReduceAverage {};
template <>
struct KokkosComm::Impl::is_reduction_operator<ReduceAverage> : public std::true_type {};

struct ReduceLogicalAnd {};
template <>
struct KokkosComm::Impl::is_reduction_operator<ReduceLogicalAnd> : public std::true_type {};

struct ReduceLogicalOr {};
template <>
struct KokkosComm::Impl::is_reduction_operator<ReduceLogicalOr> : public std::true_type {};

struct ReduceBinaryAnd {};
template <>
struct KokkosComm::Impl::is_reduction_operator<ReduceBinaryAnd> : public std::true_type {};

struct ReduceBinaryOr {};
template <>
struct KokkosComm::Impl::is_reduction_operator<ReduceBinaryOr> : public std::true_type {};

struct ReduceMaximumLoc {};
template <>
struct KokkosComm::Impl::is_reduction_operator<ReduceMaximumLoc> : public std::true_type {};

struct ReduceMinimumLoc {};
template <>
struct KokkosComm::Impl::is_reduction_operator<ReduceMinimumLoc> : public std::true_type {};

}  // namespace KokkosComm
