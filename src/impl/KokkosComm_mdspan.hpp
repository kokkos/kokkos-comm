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

#if KOKKOSCOMM_ENABLE_MDSPAN
#if KOKKOSCOMM_EXPERIMENTAL_MDSPAN
#include <experimental/mdspan>
namespace KokkosComm::Impl {
template <typename... Args> using mdspan = std::experimental::mdspan<Args...>;

template <typename IndexType, std::size_t... Extents>
using extents = std::experimental::extents<IndexType, Extents...>;

template <typename IndexType, std::size_t Rank>
using dextents = std::experimental::dextents<IndexType, Rank>;
} // namespace KokkosComm::Impl
#else
#include <mdspan>
namespace KokkosComm::Impl {
template <typename... Args> using mdspan = std::mdspan<Args...>;

template <typename IndexType, std::size_t... Extents>
using extents = std::extents<IndexType, Extents...>;

template <typename IndexType, std::size_t Rank>
using dextents = std::dextents<IndexType, Rank>;
} // namespace KokkosComm::Impl
#endif

template <typename> struct is_mdspan : std::false_type {};

template <typename... Args>
struct is_mdspan<KokkosComm::Impl::mdspan<Args...>> : std::true_type {};

template <typename... Args>
constexpr bool is_mdspan_v = is_mdspan<Args...>::value;

static_assert(is_mdspan_v<KokkosComm::Impl::mdspan<
                  float, KokkosComm::Impl::extents<size_t, 0, 3>>>,
              "");
static_assert(!is_mdspan_v<std::vector<char>>, "");

#endif // KOKKOSCOMM_ENABLE_MDSPAN