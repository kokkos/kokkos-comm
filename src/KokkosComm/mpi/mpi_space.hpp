// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <mpi.h>

#include <KokkosComm/concepts.hpp>

namespace KokkosComm {

// TODO: not sure what members this thing needs
struct Mpi {
  // TODO: just an example
  static int world_size() {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }

  // TODO: just an example
  static int world_rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

};  // struct Mpi

// KokkosComm::Mpi is a KokkosComm::CommunicationSpace
template <>
struct Impl::is_communication_space<KokkosComm::Mpi> : public std::true_type {};

}  // namespace KokkosComm
