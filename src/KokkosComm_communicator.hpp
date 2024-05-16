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

#include <cstdio>
#include <mpi.h>

namespace KokkosComm {

using Color = int;
using Key   = int;

template <KokkosExecutionSpace ExecSpace>
class Communicator {
 public:
  ~Communicator() {
    switch (_comm_kind) {
      case CommunicatorKind::User: MPI_Comm_free(&_comm); break;
      // case CommunicatorKind::Inter: MPI_Comm_disconnect(&_comm); break;
      default: break;
    }
  }

  static auto from_raw(MPI_Comm raw) -> Communicator {
    assert(MPI_COMM_NULL != raw);

    CommunicatorKind comm_kind;
    if (MPI_COMM_SELF == raw) {
      comm_kind = CommunicatorKind::Self;
    } else if (MPI_COMM_WORLD == raw) {
      comm_kind = CommunicatorKind::World;
    } else {
      int flag;
      MPI_Comm_test_inter(raw, &flag);
      if (0 != flag) {
        fprintf(stderr, "[KokkosComm] error: intercommunicators are not supported (yet).\n");
        std::terminate();
        // MPI_Comm parent_comm = MPI_COMM_NULL;
        // MPI_Comm_get_parent(&parent_comm);
        // if (raw == parent_comm) {
        //   comm_kind = CommunicatorKind::Parent;
        // } else {
        //   comm_kind = CommunicatorKind::Inter;
        // }
      } else {
        comm_kind = CommunicatorKind::User;
      }
    }

    return Communicator(raw, comm_kind);
  }

  inline static auto from_raw_unchecked(MPI_Comm comm) -> Communicator {
    return Communicator(comm, CommunicatorKind::User);
  }

  static auto dup_raw(MPI_Comm raw) -> Communicator {
    MPI_Comm new_comm;
    MPI_Comm_dup(raw, &new_comm);
    return Communicator(new_comm, CommunicatorKind::User);
  }

  static auto dup(const Communicator &other) -> Communicator { return Communicator::dup_raw(other.as_raw()); }

  static auto split_raw(MPI_Comm raw, Color color, Key key) -> Communicator {
    MPI_Comm new_comm;
    MPI_Comm_split(raw, color, key, &new_comm);
    return Communicator(new_comm, CommunicatorKind::User);
  }

  static auto split(const Communicator &other, Color color, Key key) -> Communicator {
    return Communicator::split_raw(other.as_raw(), color, key);
  }

  inline auto as_raw() const -> MPI_Comm { return _comm; }

  inline static auto self(void) -> Communicator { return Communicator::from_raw_unchecked(MPI_COMM_SELF); }

  inline static auto world(void) -> Communicator { return Communicator::from_raw_unchecked(MPI_COMM_WORLD); }

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

 private:
  enum class CommunicatorKind {
    Self,   // MPI_COMM_SELF
    World,  // MPI_COMM_WORLD
    User,   // User-defined communicator
    // Parent,
    // Inter,
  };

  Communicator(MPI_Comm comm, CommunicatorKind comm_kind) : _comm(comm), _comm_kind(comm_kind) {}

  MPI_Comm _comm;
  CommunicatorKind _comm_kind;
  ExecSpace _exec_space;
};

}  // namespace KokkosComm
