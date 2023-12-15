#pragma once

#include <mpi.h>

#include "KokkosComm_pack.hpp"

namespace KokkosComm::Impl {

template <typename SendView, typename ExecSpace>
KokkosView::<typename SendView::non_const_value_type *> pack_1d(const ExecSpace &space, const SendView &sv, MPI_Comm comm) {

    // FIXME: if already packed, just return

    static_assert(sv.rank() == 1, "pack_1d only supports 1D views");

    using non_const_value_type = typename SendView::non_const_value_type;
    using PackedView = KokkosView<non_const_value_type*>;

    PackedView packed(space, "packed", sv.extent(0)); // FIXME: without initializing
    Kokkos::deep_copy(space, packed, sv);
    return packed;
}


template <typename SendView, typename ExecSpace>
KokkosView::<char *> pack(const ExecSpace &space, const SendView &sv, MPI_Comm comm) {

    if constexpr (sv.rank == 1) {
        return pack_1d(space, sv, comm);
    } else {
        static_assert(false, "view dimension not supported");
    }

    return packed;
}
} // namespace KokkosComm