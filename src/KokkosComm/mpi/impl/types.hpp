// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#if defined(USE_CACHE)
#include <array>
#include <map>
#endif

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/mpi/mpi_space.hpp>

namespace KokkosComm::Impl {

template <KokkosView View>
MPI_Datatype view_mpi_type(const View &view) {
#define USE_CACHE

#if defined(USE_CACHE)
  using Key = std::array<int, 2 * View::rank>;
  static std::map<Key, MPI_Datatype> cache;

  Key key;
  for (size_t d = 0; d < View::rank; d++) {
    key[2 * d]     = view.extent(d);
    key[2 * d + 1] = view.stride(d);
  }
  if (cache.count(key) > 0) {
    return cache[key];
  }
#endif

  using value_type  = typename View::non_const_value_type;
  MPI_Datatype type = datatype<MpiSpace, value_type>();

  // This doesn't work for 1D contiguous views into reduce because it
  // represents the whole 1D view as 1 Hvector, rather than N elements.
  // FIXME: is there a more generic way to handle this, maybe by treating
  // the last dimension specially under certain circumstances?
  for (size_t d = 0; d < KokkosComm::rank<View>(); ++d) {
    MPI_Datatype newtype;
    MPI_Type_create_hvector(
        KokkosComm::extent(view, d) /*count*/, 1 /*block length*/, KokkosComm::stride(view, d) * sizeof(value_type),
        type, &newtype
    );
    type = newtype;
  }
  MPI_Type_commit(&type);
#if defined(USE_CACHE)
  cache[key] = type;
#endif
  return type;
}

};  // namespace KokkosComm::Impl
