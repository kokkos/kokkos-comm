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

#include "KokkosComm_concepts.hpp"
#include "KokkosComm_types.hpp"
#include "KokkosComm_include_mpi.hpp"

namespace KokkosComm::Impl {
namespace Packer {

template <KokkosView View, typename Packer>
struct MpiArgs {
  using packer_type = Packer;  // the type of the packer that produced these arguments

  View view;
  MPI_Datatype datatype;
  int count;

  MpiArgs(const View &_view, const MPI_Datatype _datatype, const int _count)
      : view(_view), datatype(_datatype), count(_count) {}
};

template <KokkosView View>
struct DeepCopy {
  using non_const_packed_view_type =
      Kokkos::View<typename View::non_const_data_type, Kokkos::LayoutRight, typename View::memory_space>;

  using args_type = MpiArgs<non_const_packed_view_type, DeepCopy<View>>;

  template <KokkosExecutionSpace ExecSpace>
  static args_type allocate_packed_for(const ExecSpace &space, const std::string &label, const View &src) {
    using KCT = KokkosComm::Traits<View>;

    if constexpr (KCT::rank() == 1) {
      non_const_packed_view_type packed(Kokkos::view_alloc(space, Kokkos::WithoutInitializing, label), src.extent(0));
      return args_type(packed, MPI_PACKED, KCT::span(packed) * sizeof(typename non_const_packed_view_type::value_type));
    } else if constexpr (KCT::rank() == 2) {
      non_const_packed_view_type packed(Kokkos::view_alloc(space, Kokkos::WithoutInitializing, label), src.extent(0),
                                        src.extent(1));
      return args_type(packed, MPI_PACKED, KCT::span(packed) * sizeof(typename non_const_packed_view_type::value_type));
    } else {
      static_assert(std::is_void_v<View>, "allocate_packed_for for rank >= 2 views unimplemented");
    }
  }

  template <KokkosExecutionSpace ExecSpace>
  static args_type pack(const ExecSpace &space, const View &src) {
    args_type args = allocate_packed_for(space, "DeepCopy::pack", src);
    Kokkos::deep_copy(space, args.view, src);
    return args;
  }

  template <KokkosExecutionSpace ExecSpace>
  static void unpack_into(const ExecSpace &space, const View &dst, const non_const_packed_view_type &src) {
    Kokkos::deep_copy(space, dst, src);
  }
};

template <KokkosView View>
struct MpiDatatype {
  using non_const_packed_view_type = View;
  using args_type                  = MpiArgs<non_const_packed_view_type, MpiDatatype<View>>;

  // don't actually allocate - return the provided view, but with
  // a datatype that describes the data in the view
  template <KokkosExecutionSpace ExecSpace>
  static args_type allocate_packed_for(const ExecSpace & /*space*/, const std::string & /*label*/, const View &src) {
    using ValueType = typename View::value_type;

    using KCT = KokkosComm::Traits<View>;

    MPI_Datatype type = mpi_type<ValueType>();
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
}  // namespace KokkosComm::Impl
