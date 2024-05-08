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

#include "KokkosComm_include_mpi.hpp"

#include <Kokkos_Core.hpp>

namespace KokkosComm::Impl {
template <typename Scalar>
MPI_Datatype mpi_type() {
  static_assert(std::is_void_v<Scalar>, "mpi_type not implemented");
  return MPI_CHAR;  // unreachable
}

template <>
inline MPI_Datatype mpi_type<int>() {
  return MPI_INT;
}
template <>
inline MPI_Datatype mpi_type<long int>() {
  return MPI_LONG;
}
template <>
inline MPI_Datatype mpi_type<long long>() {
  return MPI_LONG_LONG;
}

template <>
inline MPI_Datatype mpi_type<unsigned int>() {
  return MPI_UNSIGNED;
}
template <>
inline MPI_Datatype mpi_type<unsigned long>() {
  return MPI_UNSIGNED_LONG;
}
template <>
inline MPI_Datatype mpi_type<unsigned long>() {
  return MPI_UNSIGNED_LONG_LONG;
}

template <>
inline MPI_Datatype mpi_type<float>() {
  return MPI_FLOAT;
}
template <>
inline MPI_Datatype mpi_type<double>() {
  return MPI_DOUBLE;
}
template <>
inline MPI_Datatype mpi_type<long double>() {
  return MPI_LONG_DOUBLE;
}

template <>
inline MPI_Datatype mpi_type<Kokkos::complex<float>>() {
  return MPI_COMPLEX;
}
template <>
inline MPI_Datatype mpi_type<Kokkos::complex<double>>() {
  return MPI_DOUBLE_COMPLEX;
}

template <>
inline MPI_Datatype mpi_type<int8_t>() {
  return MPI_INT8_T;
}
template <>
inline MPI_Datatype mpi_type<int16_t>() {
  return MPI_INT16_T;
}
template <>
inline MPI_Datatype mpi_type<int32_t>() {
  return MPI_INT32_T;
}
template <>
inline MPI_Datatype mpi_type<int64_t>() {
  return MPI_INT64_T;
}

template <>
inline MPI_Datatype mpi_type<uint8_t>() {
  return MPI_UINT8_T;
}
template <>
inline MPI_Datatype mpi_type<uint16_t>() {
  return MPI_UINT16_T;
}
template <>
inline MPI_Datatype mpi_type<uint32_t>() {
  return MPI_UINT32_T;
}
template <>
inline MPI_Datatype mpi_type<uint64_t>() {
  return MPI_UINT64_T;
}

template <>
inline MPI_Datatype mpi_type<size_t>() {
  if constexpr ( sizeof(size_t) == 1 ) return MPI_UINT8_T;
  if constexpr ( sizeof(size_t) == 2 ) return MPI_UINT16_T;
  if constexpr ( sizeof(size_t) == 4 ) return MPI_UINT32_T;
  if constexpr ( sizeof(size_t) == 8 ) return MPI_UINT64_T;
}
template <>
inline MPI_Datatype mpi_type<ssize_t>() {
  if constexpr ( sizeof(ssize_t) == 1 ) return MPI_INT8_T;
  if constexpr ( sizeof(ssize_t) == 2 ) return MPI_INT16_T;
  if constexpr ( sizeof(ssize_t) == 4 ) return MPI_INT32_T;
  if constexpr ( sizeof(ssize_t) == 8 ) return MPI_INT64_T;
}

template <typename Scalar>
inline MPI_Datatype mpi_type_v = mpi_type<Scalar>();

};  // namespace KokkosComm::Impl
