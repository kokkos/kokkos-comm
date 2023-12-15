#pragma once

#include <mpi.h>

#include <Kokkos_Core.hpp>

namespace KokkosComm::Impl {

template <typename DstView, typename SrcView, typename ExecSpace>
void unpack_into_1d(const ExecSpace &space, const DstView &dst,
                    const SrcView &src) {

  // FIXME: if already unpacked, just return

  static_assert(DstView::rank == 1, "unpack_into_1d only supports 1D views");

  Kokkos::deep_copy(space, dst, src);
}

template <typename DstView, typename SrcView, typename ExecSpace>
void unpack_into(const ExecSpace &space, const DstView &dst,
                 const SrcView &src) {

  if constexpr (DstView::rank == 1) {
    unpack_into_1d(space, dst, src);
  } else {
    static_assert(std::is_void_v<DstView>,
                  "unpack_into view dimension not supported");
  }
}
} // namespace KokkosComm::Impl