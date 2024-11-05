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

#include <KokkosComm/mpi/impl/include_mpi.hpp>

#include <Kokkos_Core.hpp>

namespace KokkosComm::Impl {
template <typename Scalar>
MPI_Datatype mpi_type() {
  using T = std::decay_t<Scalar>;

  if constexpr (std::is_same_v<T, std::byte>)
    return MPI_BYTE;

  else if constexpr (std::is_same_v<T, char>)
    return MPI_CHAR;
  else if constexpr (std::is_same_v<T, unsigned char>)
    return MPI_UNSIGNED_CHAR;

  else if constexpr (std::is_same_v<T, short>)
    return MPI_SHORT;
  else if constexpr (std::is_same_v<T, unsigned short>)
    return MPI_UNSIGNED_SHORT;

  else if constexpr (std::is_same_v<T, int>)
    return MPI_INT;
  else if constexpr (std::is_same_v<T, unsigned>)
    return MPI_UNSIGNED;

  else if constexpr (std::is_same_v<T, long>)
    return MPI_LONG;
  else if constexpr (std::is_same_v<T, unsigned long>)
    return MPI_UNSIGNED_LONG;

  else if constexpr (std::is_same_v<T, long long>)
    return MPI_LONG_LONG;
  else if constexpr (std::is_same_v<T, unsigned long long>)
    return MPI_UNSIGNED_LONG_LONG;

  else if constexpr (std::is_same_v<T, std::int8_t>)
    return MPI_INT8_T;
  else if constexpr (std::is_same_v<T, std::uint8_t>)
    return MPI_UINT8_T;

  else if constexpr (std::is_same_v<T, std::int16_t>)
    return MPI_INT16_T;
  else if constexpr (std::is_same_v<T, std::uint16_t>)
    return MPI_UINT16_T;

  else if constexpr (std::is_same_v<T, std::int32_t>)
    return MPI_INT32_T;
  else if constexpr (std::is_same_v<T, std::uint32_t>)
    return MPI_UINT32_T;

  else if constexpr (std::is_same_v<T, std::int64_t>)
    return MPI_INT64_T;
  else if constexpr (std::is_same_v<T, std::uint64_t>)
    return MPI_UINT64_T;

  else if constexpr (std::is_same_v<T, std::size_t>) {
    if constexpr (sizeof(std::size_t) == 1) return MPI_UINT8_T;
    if constexpr (sizeof(std::size_t) == 2) return MPI_UINT16_T;
    if constexpr (sizeof(std::size_t) == 4) return MPI_UINT32_T;
    if constexpr (sizeof(std::size_t) == 8) return MPI_UINT64_T;
  }

  else if constexpr (std::is_same_v<T, std::ptrdiff_t>) {
    if constexpr (sizeof(std::ptrdiff_t) == 1) return MPI_INT8_T;
    if constexpr (sizeof(std::ptrdiff_t) == 2) return MPI_INT16_T;
    if constexpr (sizeof(std::ptrdiff_t) == 4) return MPI_INT32_T;
    if constexpr (sizeof(std::ptrdiff_t) == 8) return MPI_INT64_T;
  }

  else if constexpr (std::is_same_v<T, float>)
    return MPI_FLOAT;
  else if constexpr (std::is_same_v<T, double>)
    return MPI_DOUBLE;
  else if constexpr (std::is_same_v<T, long double>)
    return MPI_LONG_DOUBLE;

  else if constexpr (std::is_same_v<T, Kokkos::complex<float>>)
    return MPI_COMPLEX;
  else if constexpr (std::is_same_v<T, Kokkos::complex<double>>)
    return MPI_DOUBLE_COMPLEX;

  else {
    static_assert(std::is_void_v<T>, "mpi_type not implemented");
    return MPI_CHAR;  // unreachable
  }
}

template <typename Scalar>
inline MPI_Datatype mpi_type_v = mpi_type<Scalar>();

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
  MPI_Datatype type = mpi_type_v<value_type>;

  // This doesn't work for 1D contiguous views into reduce because it
  // represents the whole 1D view as 1 Hvector, rather than N elements.
  // FIXME: is there a more generic way to handle this, maybe by treating
  // the last dimension specially under certain circumstances?
  for (size_t d = 0; d < KokkosComm::rank<View>(); ++d) {
    MPI_Datatype newtype;
    MPI_Type_create_hvector(KokkosComm::extent(view, d) /*count*/, 1 /*block length*/,
                            KokkosComm::stride(view, d) * sizeof(value_type), type, &newtype);
    type = newtype;
  }
  MPI_Type_commit(&type);
#if defined(USE_CACHE)
  cache[key] = type;
#endif
  return type;
}

};  // namespace KokkosComm::Impl
