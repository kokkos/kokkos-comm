#pragma once

#include <mpi.h>

namespace KokkosComm::Impl {
template <typename Scalar> struct mpi_type {};

template <> struct mpi_type<double> {
  static constexpr MPI_Datatype value = MPI_DOUBLE;
};

template <typename Scalar>
inline constexpr MPI_Datatype mpi_type_v = mpi_type<Scalar>::value;

}; // namespace KokkosComm::Impl