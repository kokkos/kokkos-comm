#pragma once

namespace KokkosComm::Impl {

template <typename View> struct packed_view {
  using type = Kokkos::View<typename View::non_const_data_type,
                            typename View::memory_space>;
};

template <typename View>
typename packed_view<View>::type allocate_packed(const std::string &label,
                                                 const View &v) {

  if constexpr (View::rank == 1) {
    return typename packed_view<View>::type(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, label), v.extent(0));
  } else {
    static_assert(std::is_void_v<>,
                  "allocate_packed only supports rank-1 views");
  }
}

} // namespace KokkosComm::Impl