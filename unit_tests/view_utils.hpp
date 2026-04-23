// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include <Kokkos_Core.hpp>

namespace test_utils {

struct Contig {};
struct NonContig {};

namespace Impl {

template <typename T, size_t N>
struct AddPtrs {
  using type = typename AddPtrs<T*, N - 1>::type;
};
template <typename T>
struct AddPtrs<T, 0> {
  using type = T;
};
template <typename T, size_t N>
using Stars = typename AddPtrs<T, N>::type;

template <typename View, size_t... Is>
struct InitFunctor {
  View v;
  std::array<int, sizeof...(Is)> exts;

  KOKKOS_FUNCTION void operator()(decltype(static_cast<int>(std::declval<View>().extent(Is)))... idxs) const {
    int val   = 0;
    int ids[] = {int(idxs)...};
    ((val = ids[Is] + exts[Is] * val), ...);
    [&]<size_t... Js>(std::index_sequence<Js...>) { v(ids[Js]...) = val; }
    (std::index_sequence<Is...>{});
  }
};

template <typename View, size_t... Is>
struct CountFunctor {
  using Scalar = typename View::value_type;
  View v;
  std::array<int, sizeof...(Is)> exts;

  KOKKOS_FUNCTION void operator()(decltype(static_cast<int>(std::declval<View>().extent(Is)))... idxs, int& lsum)
      const {
    int val   = 0;
    int ids[] = {int(idxs)...};
    ((val = ids[Is] + exts[Is] * val), ...);
    [&]<size_t... Js>(std::index_sequence<Js...>) { lsum += v(ids[Js]...) != Scalar(val); }
    (std::index_sequence<Is...>{});
  }
};

}  // namespace Impl

template <typename T, size_t R, typename... Extents>
auto build_view(Contig, const std::string& name, Extents... exts) {
  static_assert(sizeof...(exts) == R, "Number of extents must match Rank");
  return Kokkos::View<Impl::Stars<T, R>>(name, exts...);
}
template <typename T, size_t R, typename... Extents>
auto build_view(NonContig, const std::string& name, Extents... exts) {
  static_assert(sizeof...(exts) == R, "Number of extents must match Rank");
  Kokkos::View<Impl::Stars<T, R + 1>, Kokkos::LayoutRight> v(name, exts..., 2);
  return [&]<size_t... Is>(std::index_sequence<Is...>) { return Kokkos::subview(v, (void(Is), Kokkos::ALL)..., 1); }
  (std::make_index_sequence<R>{});
}

template <typename Exec, typename View>
auto init_view(const Exec& exec, const View& v) -> void {
  constexpr size_t R = View::rank;
  if constexpr (R == 1) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Exec>(exec, 0, v.extent(0)), KOKKOS_LAMBDA(const int i) { v(i) = i; }
    );
  } else {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      std::array<int, R> exts = {static_cast<int>(v.extent(Is))...};
      Kokkos::parallel_for(
          Kokkos::MDRangePolicy<Exec, Kokkos::Rank<R>>(exec, {(void(Is), 0)...}, {static_cast<int>(v.extent(Is))...}),
          Impl::InitFunctor<View, Is...>{v, exts}
      );
    }
    (std::make_index_sequence<R>{});
  }
  exec.fence();
}

template <typename View>
auto count_errors(const View& v) -> int {
  constexpr size_t R = View::rank;
  int errs           = 0;
  using Scalar       = typename View::value_type;
  if constexpr (R == 1) {
    Kokkos::parallel_reduce(
        v.extent(0), KOKKOS_LAMBDA(const int i, int& lsum) { lsum += v(i) != Scalar(i); }, errs
    );
  } else {
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      std::array<int, R> exts = {static_cast<int>(v.extent(Is))...};
      Kokkos::parallel_reduce(
          Kokkos::MDRangePolicy<Kokkos::Rank<R>>({(void(Is), 0)...}, {static_cast<int>(v.extent(Is))...}),
          Impl::CountFunctor<View, Is...>{v, exts}, errs
      );
    }
    (std::make_index_sequence<R>{});
  }
  return errs;
}

}  // namespace test_utils
