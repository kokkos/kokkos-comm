// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>

#include <KokkosComm/impl/contiguous.hpp>
#include "types.hpp"

namespace KokkosComm::Experimental::nccl::Impl::Packer {

template <KokkosView View>
struct PackedNcclView {
  View view_;
  ncclDataType_t datatype_;
  int count_;

  PackedNcclView(const View &view, const ncclDataType_t datatype, const int count)
      : view_(view), datatype_(datatype), count_(count) {}
};

template <KokkosView View>
struct DeepCopy {
  using PackedView = KokkosComm::Impl::contiguous_view_t<View>;

  template <KokkosExecutionSpace ExecSpace>
  static auto pack(const ExecSpace &space, const View &src) -> PackedNcclView<PackedView> {
    PackedView packed_src = KokkosComm::Impl::allocate_contiguous_for(space, "DeepCopy::pack", src);
    // Use `ncclUint8` because there is no equivalent to `MPI_PACKED`.
    PackedNcclView<PackedView> packed(packed_src, ncclUint8, src.size() * sizeof(typename PackedView::value_type));
    Kokkos::deep_copy(space, packed.view_, src);
    return packed;
  }

  template <KokkosExecutionSpace ExecSpace>
  static auto unpack_into(const ExecSpace &space, View &dst, const PackedView &src) -> void {
    Kokkos::deep_copy(space, dst, src);
  }
};

}  // namespace KokkosComm::Experimental::nccl::Impl::Packer
