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
