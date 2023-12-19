// Copyright 2023 Carl Pearson
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mpi.h>

#include <Kokkos_Core.hpp>

namespace KokkosComm::Impl {
template <typename Scalar> MPI_Datatype mpi_type() {
  static_assert(std::is_void_v<Scalar>, "mpi_type not implemented");
  return MPI_CHAR; // unreachable
};

template <> inline MPI_Datatype mpi_type<unsigned int>() {
  return MPI_UNSIGNED;
}
template <> inline MPI_Datatype mpi_type<unsigned long>() {
  return MPI_UNSIGNED_LONG;
}
template <> inline MPI_Datatype mpi_type<long int>() { return MPI_LONG; }
template <> inline MPI_Datatype mpi_type<long long>() { return MPI_LONG_LONG; }
template <> inline MPI_Datatype mpi_type<int>() { return MPI_INT; }
template <> inline MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype mpi_type<float>() { return MPI_FLOAT; }
template <> inline MPI_Datatype mpi_type<Kokkos::complex<float>>() {
  return MPI_COMPLEX;
}
template <> inline MPI_Datatype mpi_type<Kokkos::complex<double>>() {
  return MPI_DOUBLE_COMPLEX;
}

template <typename Scalar> inline MPI_Datatype mpi_type_v = mpi_type<Scalar>();

}; // namespace KokkosComm::Impl