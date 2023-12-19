#pragma once

#include <mpi.h>

namespace KokkosComm::Impl {
template <typename Scalar> struct mpi_type {};

template <> struct mpi_type<double> {
  static const MPI_Datatype value = MPI_DOUBLE;
};

template <typename Scalar>
inline MPI_Datatype mpi_type_v = mpi_type<Scalar>::value;

}; // namespace KokkosComm::Impl