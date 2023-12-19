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

namespace KokkosComm {

/*! \brief This can be specialized to do custom behavior for a particular view*/
template <typename View> struct Traits {
  static bool needs_unpack(const View &v) { return !v.span_is_contiguous(); }
  static bool needs_pack(const View &v) { return !v.span_is_contiguous(); }

  using non_const_packed_view_type =
      Kokkos::View<typename View::non_const_data_type,
                   typename View::array_layout, typename View::memory_space>;
  using packed_view_type =
      Kokkos::View<typename View::data_type, typename View::array_layout,
                   typename View::memory_space>;

  template <typename ExecSpace>
  static void pack(const ExecSpace &space,
                   const non_const_packed_view_type &dst, const View &src) {
    Kokkos::deep_copy(space, dst, src);
  }

  template <typename ExecSpace>
  static void unpack(const ExecSpace &space,
                     const non_const_packed_view_type &dst, const View &src) {
    Kokkos::deep_copy(space, dst, src);
  }
};
} // namespace KokkosComm
