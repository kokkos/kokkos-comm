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

#include "KokkosComm_concepts.hpp"

#include <Kokkos_Core_fwd.hpp>
#include <cstdio>
#include <mpi.h>

namespace KokkosComm {

using Color = int;
using Key   = int;

template <KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace>
class Communicator {
 private:
  MPI_Comm _comm;
  ExecSpace _exec_space;

 public:
  Communicator(MPI_Comm comm) : _comm(comm) {}
  Communicator(const Communicator& other) = delete;
  Communicator(const Communicator&& other) { _comm = std::move(other._comm); }
  ~Communicator() {
    // Only free the communicator if it hasn't been set to `MPI_COMM_NULL` before. This is to prevent double freeing
    // when we explicitly call the communicator's dtor in the `Context` dtor.
    if (MPI_COMM_NULL != _comm) {
      MPI_Comm_free(&_comm);
    }
  }

  static auto dup_raw(MPI_Comm raw) -> Communicator {
    MPI_Comm new_comm;
    MPI_Comm_dup(raw, &new_comm);
    return Communicator(new_comm);
  }

  static auto dup(const Communicator& other) -> Communicator { return Communicator::dup_raw(other.as_raw()); }

  static auto split_raw(MPI_Comm raw, Color color, Key key) -> Communicator {
    MPI_Comm new_comm;
    MPI_Comm_split(raw, color, key, &new_comm);
    return Communicator(new_comm);
  }

  static auto split(const Communicator& other, Color color, Key key) -> Communicator {
    return Communicator::split_raw(other.as_raw(), color, key);
  }

  inline auto as_raw() const -> MPI_Comm { return _comm; }

  inline auto size(void) const -> int {
    int size;
    MPI_Comm_size(_comm, &size);
    return size;
  }

  inline auto rank(void) const -> int {
    int rank;
    MPI_Comm_rank(_comm, &rank);
    return rank;
  }
};

}  // namespace KokkosComm
