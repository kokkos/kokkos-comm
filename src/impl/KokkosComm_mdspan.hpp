// Copyright 2023 Carl Pearson
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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