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

#include <type_traits>

#include "KokkosComm_concepts.hpp"
#include "impl/KokkosComm_include_mpi.hpp"

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

// KokkosComm::Mpi is a KokkosComm::Transport
template <>
struct Impl::is_transport<KokkosComm::Mpi> : public std::true_type {};

}  // namespace KokkosComm