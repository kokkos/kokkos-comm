// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <string>

#include <mpi.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>

#include <KokkosComm/impl/contiguous.hpp>

namespace KokkosComm::mpi::Impl {
namespace Packer {

template <KokkosView View>
struct MpiArgs {
  View view;
  MPI_Datatype datatype;
  int count;

  MpiArgs(const View &_view, const MPI_Datatype _datatype, const int _count)
      : view(_view), datatype(_datatype), count(_count) {}
};

template <KokkosView V>
struct DeepCopy {
  using PackedV = KokkosComm::Impl::contiguous_view_t<V>;
  using Args    = MpiArgs<PackedV>;
  using T       = typename PackedV::non_const_value_type;

  /// Returns allocated, uninitialized, contiguous view for packing `src`.
  template <KokkosExecutionSpace ES>
  static auto allocate_packed_for(const ES &space, const std::string &label, const V &src) -> Args {
    auto packed = KokkosComm::Impl::allocate_contiguous_for(space, label, src);
    return Args(packed, datatype<MpiSpace, T>(), span(packed));
  }

  /// Returns packed view from `src`.
  template <KokkosExecutionSpace ES>
  static auto pack(const ES &space, const std::string &label, const V &src) -> Args {
    auto args = allocate_packed_for(space, label, src);
    Kokkos::deep_copy(space, args.view, src);
    return args;
  }

  /// Unpacks `src` view into `dst`.
  template <KokkosExecutionSpace ES>
  static auto unpack_into(const ES &space, const V &dst, const PackedV &src) -> void {
    Kokkos::deep_copy(space, dst, src);
  }
};

template <KokkosView View>
struct MpiDatatype {
  using non_const_packed_view_type = View;
  using args_type                  = MpiArgs<non_const_packed_view_type>;

  // don't actually allocate - return the provided view, but with
  // a datatype that describes the data in the view
  template <KokkosExecutionSpace ExecSpace>
  static args_type allocate_packed_for(const ExecSpace & /*space*/, const std::string & /*label*/, const View &src) {
    using ValueType = typename View::value_type;
    using KCT       = KokkosComm::Traits<View>;

    MPI_Datatype type = datatype<MpiSpace, ValueType>()();
    for (size_t d = 0; d < KokkosComm::Traits<View>::rank(); ++d) {
      MPI_Datatype newtype;
      MPI_Type_create_hvector(KCT::extent(src, d) /*count*/, 1 /*block length*/,
                              KCT::stride(src, d) * sizeof(ValueType), type, &newtype);
      type = newtype;
    }
    MPI_Type_commit(&type);
    return args_type(src, type, 1);
  }

  // pack is a no-op: rely on MPI's datatype engine
  template <KokkosExecutionSpace ExecSpace>
  static args_type pack(const ExecSpace &space, const View &src) {
    return allocate_packed_for(space, "", src);
  }

  // unpack is a no-op: rely on MPI's datatype engine
  template <KokkosExecutionSpace ExecSpace>
  static void unpack_into(const ExecSpace & /*space*/, const View & /*dst*/,
                          const non_const_packed_view_type & /*src*/) {
    return;
  }
};

}  // namespace Packer
}  // namespace KokkosComm::mpi::Impl
