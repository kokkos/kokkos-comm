//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#pragma once

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/impl/KokkosComm_contiguous.hpp>
#include <KokkosComm/nccl/impl/types.hpp>

#include <nccl.h>

namespace KokkosComm::Experimental::nccl::Impl::Packer {

template <KokkosView View>
struct NcclArgs {
  View view_;
  ncclDataType_t datatype_;
  int count_;

  NcclArgs(const View &view, const ncclDataType_t datatype, const int count)
      : view_(view), datatype_(datatype), count_(count) {}
};

template <KokkosView View>
struct DeepCopy {
  using PackedView = KokkosComm::Impl::contiguous_view_t<View>;
  using Args       = NcclArgs<PackedView>;

  template <KokkosExecutionSpace ExecSpace>
  static auto pack(const ExecSpace &space, const View &src) -> Args {
    PackedView packed_src = KokkosComm::Impl::allocate_contiguous_for(space, "DeepCopy::pack", src);
    // Use `ncclUint8` because there is no equivalent to `MPI_PACKED`.
    Args args(packed_src, ncclUint8, src.span() * sizeof(PackedView::value_type));
    Kokkos::deep_copy(space, args.view, src);
    return args;
  }

  template <KokkosExecutionSpace ExecSpace>
  static auto unpack_into(const ExecSpace &space, const View &dst, const PackedView &src) -> void {
    Kokkos::deep_copy(space, dst, src);
  }
};

}  // namespace KokkosComm::Experimental::nccl::Impl::Packer
