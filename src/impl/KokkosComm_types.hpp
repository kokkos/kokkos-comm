#pragma once

#include <mpi.h>

#include <Kokkos_Core.hpp>

namespace KokkosComm::Impl {
template <typename Scalar> MPI_Datatype mpi_type() {
  static_assert(std::is_void_v<Scalar>, "mpi_type not implemented");
};

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