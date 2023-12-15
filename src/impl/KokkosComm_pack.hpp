#pragma once

#include <mpi.h>

#include <Kokkos_Core.hpp>

namespace KokkosComm::Impl {

template <typename SendView, typename ExecSpace>
Kokkos::View<typename SendView::non_const_value_type *>
pack_1d(const ExecSpace &space, const SendView &sv, MPI_Comm comm) {

  // FIXME: if already packed, just return

  static_assert(SendView::rank == 1, "pack_1d only supports 1D views");

  using non_const_value_type = typename SendView::non_const_value_type;
  using PackedView = Kokkos::View<non_const_value_type *>;

  PackedView packed(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "packed"),
      sv.extent(0));
  Kokkos::deep_copy(space, packed, sv);
  return packed;
}

template <typename SendView, typename ExecSpace>
Kokkos::View<typename SendView::non_const_value_type *>
pack(const ExecSpace &space, const SendView &sv, MPI_Comm comm) {

  if constexpr (SendView::rank == 1) {
    return pack_1d(space, sv, comm);
  } else {
    static_assert(std::is_void_v<SendView>, "view dimension not supported");
  }
}
} // namespace KokkosComm::Impl