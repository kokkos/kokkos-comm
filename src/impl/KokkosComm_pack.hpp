#pragma once

#include <mpi.h>

#include "KokkosComm_pack.hpp"

namespace KokkosComm::Impl {
template <typename SendView, typename ExecSpace>
KokkosView::<char *> pack(const ExecSpace &space, const SendView &sv, MPI_Comm comm) {
    using value_type = typename SendView::value_type;
    using PackedView = KokkosView<char*>;

    // determine the total packed size
    const int packedSize = sizeof(value_type) * sv.extent(0);
    
    KokkosView<char *> packed(space, "packed", packedSize);

    Kokkos::parallel_for(space, sv.extent(0), KOKKOS_LAMBDA(const int si){
        const size_t packedOffset = si * sizeof(value_type);
        auto sv = packed::subview(packedOffset, packedOffset+sizeof(value_type));
        Kokkos::deep_copy(sv, sv(si));
    });

    return packed;
}
} // namespace KokkosComm