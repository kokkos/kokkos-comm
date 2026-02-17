// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <string>

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>

#include <KokkosComm/impl/contiguous.hpp>

namespace KokkosComm::Experimental::nccl::Impl::Packer {

template <KokkosView View>
struct PackedNcclView {
  View view_;
  ncclDataType_t datatype_;
  int count_;

  PackedNcclView(const View& view, ncclDataType_t datatype, int count)
      : view_(view), datatype_(datatype), count_(count) {}
};

template <KokkosView V>
struct DeepCopy {
  using PackedV = KokkosComm::Impl::contiguous_view_t<V>;
  using T       = typename PackedV::non_const_value_type;

  /// @brief Allocates an uninitialized, contiguous view for packing `view`.
  /// @tparam E A Kokkos `ExecutionSpace` type.
  /// @param exec The execution space in which to perform the allocation operation.
  /// @param label The label to give to the packed view.
  /// @param view The view to allocate a packed view for.
  /// @return An allocated, uninitialized, contiguous view fit for packing `view`.
  template <KokkosExecutionSpace E>
  static auto allocate_packed_for(const E& exec, const std::string& label, const V& view) -> PackedNcclView<PackedV> {
    auto packed = KokkosComm::Impl::allocate_contiguous_for(exec, label, view);
    return PackedNcclView<PackedV>(packed, datatype<NcclSpace, T>(), span(packed));
  }

  /// @brief Packs `view` into a contiguous view.
  /// @tparam E A Kokkos `ExecutionSpace` type.
  /// @param exec The execution space in which to perform the packing operation.
  /// @param label The label to give to the packed view.
  /// @param view The view to pack.
  /// @return A packed view from `view`.
  template <KokkosExecutionSpace E>
  static auto pack(const E& exec, const std::string& label, const V& view) -> PackedNcclView<PackedV> {
    auto packed = allocate_packed_for(exec, label, view);
    Kokkos::deep_copy(exec, packed.view_, view);
    return packed;
  }

  /// @brief Unpacks `src` view into `dst`.
  /// @tparam E A Kokkos `ExecutionSpace` type.
  /// @param exec The execution space in which to perform the packing operation.
  /// @param dst The view to unpack into.
  /// @param src The packed view to unpack from.
  template <KokkosExecutionSpace E>
  static auto unpack_into(const E& exec, const V& dst, const PackedV& src) -> void {
    Kokkos::deep_copy(exec, dst, src);
  }
};

}  // namespace KokkosComm::Experimental::nccl::Impl::Packer
