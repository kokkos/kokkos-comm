// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <mpi.h>
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <nccl.h>
#endif

#include "concepts.hpp"
#include "mpi/mpi_space.hpp"
#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include "nccl/nccl_space.hpp"
#endif

namespace KokkosComm {
namespace Impl {

template <typename T>
constexpr auto mpi_datatype() -> MPI_Datatype {
  if constexpr (std::is_same_v<T, std::byte>) {
    return MPI_BYTE;
  } else if constexpr (std::is_same_v<T, char>) {
    return MPI_CHAR;
  } else if constexpr (std::is_same_v<T, unsigned char>) {
    return MPI_UNSIGNED_CHAR;
  } else if constexpr (std::is_same_v<T, short>) {
    return MPI_SHORT;
  } else if constexpr (std::is_same_v<T, unsigned short>) {
    return MPI_UNSIGNED_SHORT;
  } else if constexpr (std::is_same_v<T, int>) {
    return MPI_INT;
  } else if constexpr (std::is_same_v<T, unsigned>) {
    return MPI_UNSIGNED;
  } else if constexpr (std::is_same_v<T, long>) {
    return MPI_LONG;
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return MPI_UNSIGNED_LONG;
  } else if constexpr (std::is_same_v<T, long long>) {
    return MPI_LONG_LONG;
  } else if constexpr (std::is_same_v<T, unsigned long long>) {
    return MPI_UNSIGNED_LONG_LONG;
  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    return MPI_INT8_T;
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    return MPI_UINT8_T;
  } else if constexpr (std::is_same_v<T, std::int16_t>) {
    return MPI_INT16_T;
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return MPI_UINT16_T;
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return MPI_INT32_T;
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return MPI_UINT32_T;
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return MPI_INT64_T;
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return MPI_UINT64_T;
  } else if constexpr (std::is_same_v<T, std::size_t>) {
    if constexpr (sizeof(std::size_t) == 1) return MPI_UINT8_T;
    if constexpr (sizeof(std::size_t) == 2) return MPI_UINT16_T;
    if constexpr (sizeof(std::size_t) == 4) return MPI_UINT32_T;
    if constexpr (sizeof(std::size_t) == 8) return MPI_UINT64_T;
  } else if constexpr (std::is_same_v<T, std::ptrdiff_t>) {
    if constexpr (sizeof(std::ptrdiff_t) == 1) return MPI_INT8_T;
    if constexpr (sizeof(std::ptrdiff_t) == 2) return MPI_INT16_T;
    if constexpr (sizeof(std::ptrdiff_t) == 4) return MPI_INT32_T;
    if constexpr (sizeof(std::ptrdiff_t) == 8) return MPI_INT64_T;
  } else if constexpr (std::is_same_v<T, float>) {
    return MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return MPI_DOUBLE;
  } else if constexpr (std::is_same_v<T, long double>) {
    return MPI_LONG_DOUBLE;
  } else if constexpr (std::is_same_v<T, Kokkos::complex<float>>) {
#if defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI)
    return MPI_CXX_COMPLEX;
#else
    return MPI_COMPLEX;
#endif
  } else if constexpr (std::is_same_v<T, Kokkos::complex<double>>) {
#if defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI)
    return MPI_CXX_DOUBLE_COMPLEX;
#else
    return MPI_DOUBLE_COMPLEX;
#endif
  } else {
    static_assert(std::is_void_v<T>, "KokkosComm::Impl::mpi_datatype: datatype not implemented");
    return MPI_CHAR;  // unreachable
  }
}

#if defined(KOKKOSCOMM_ENABLE_NCCL)
template <typename T>
constexpr auto nccl_datatype() -> ncclDataType_t {
  if constexpr (std::is_same_v<T, char>) {
    return ncclChar;
  } else if constexpr (std::is_same_v<T, int>) {
    return ncclInt;
  } else if constexpr (std::is_same_v<T, unsigned> and sizeof(unsigned) == 4) {
    return ncclUint32;
  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    return ncclInt8;
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    return ncclUint8;
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return ncclInt32;
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return ncclUint32;
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return ncclInt64;
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return ncclUint64;
  } else if constexpr (std::is_same_v<T, std::size_t>) {
    if constexpr (sizeof(std::size_t) == 1) return ncclUint8;
    if constexpr (sizeof(std::size_t) == 4) return ncclUint32;
    if constexpr (sizeof(std::size_t) == 8) return ncclUint64;
  } else if constexpr (std::is_same_v<T, std::ptrdiff_t>) {
    if constexpr (sizeof(std::ptrdiff_t) == 1) return ncclInt8;
    if constexpr (sizeof(std::ptrdiff_t) == 4) return ncclInt32;
    if constexpr (sizeof(std::ptrdiff_t) == 8) return ncclInt64;
  } else if constexpr (std::is_same_v<T, float>) {
    return ncclFloat;
  } else if constexpr (std::is_same_v<T, double>) {
    return ncclDouble;
  } else {
    static_assert(std::is_void_v<T>, "KokkosComm::Impl::nccl_datatype: datatype not implemented");
    return ncclChar;  // unreachable
  }
}
#endif

}  // namespace Impl

/// @brief Converts a C++-native data type to its communication space equivalent representation.
///
/// When `C` is:
/// - `MpiSpace`, returns the corresponding `MPI_Datatype` type.
/// - `NcclSpace`, returns the corresponding `ncclDataType_t` type.
///
/// Non-system data types (i.e. the data types not natively supported by `C`) are not convertible. This notably
/// includes user-defined types.
///
/// @tparam C The target communication space backend to use for data type conversion.
/// @tparam T The C++-native data type to convert.
/// @returns The communication space representation of the C++-native data type.
template <CommunicationSpace C, typename T>
[[nodiscard]] constexpr auto datatype() -> typename C::datatype_type {
  if constexpr (std::is_same_v<C, MpiSpace>) {
    return Impl::mpi_datatype<std::remove_cv_t<T>>();
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  } else if constexpr (std::is_same_v<C, Experimental::NcclSpace>) {
    return Impl::nccl_datatype<std::remove_cv_t<T>>();
#endif
  } else {
    static_assert(std::is_void_v<C>, "KokkosComm::datatype: conversion not implemented for this communication space");
    return Impl::mpi_datatype<std::remove_cv_t<T>>();  // unreachable
  }
}

/// @returns The communication space representation of the Kokkos View value type.
template <CommunicationSpace C, KokkosView V>
[[nodiscard]] constexpr auto datatype_for([[maybe_unused]] const V& view) -> typename C::datatype_type {
  return datatype<C, std::remove_cvref_t<typename V::value_type>>();
}

/// @returns The communication space representation of the Kokkos View value type.
template <CommunicationSpace C, KokkosView V>
[[nodiscard]] constexpr auto datatype_for([[maybe_unused]] C&& comm, [[maybe_unused]] const V& view) ->
    typename C::datatype_type {
  return datatype<C, std::remove_cvref_t<typename V::value_type>>();
}

}  // namespace KokkosComm
