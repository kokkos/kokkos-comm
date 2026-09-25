//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2025) National Technology & Engineering
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

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/traits.hpp>

#include <KokkosComm/datatype.hpp>

namespace KokkosComm {

template <typename ViewType, typename CommSpace = DefaultCommunicationSpace>
class Window {
 public:
  enum class LockType { Shared, Exclusive };

  explicit Window(ViewType v, MPI_Comm comm) : v_(v), comm_(comm) {
    MPI_Win_create(
        v_.data(), v_.span() * sizeof(typename ViewType::value_type), sizeof(typename ViewType::value_type),
        MPI_INFO_NULL, comm_, &win
    );
  }

  ~Window() { MPI_Win_free(&win); }

  // Data Transfer Functions:
  template <typename T>
  void put(const T* origin_addr, int count, int target_rank, MPI_Aint target_disp) {
    MPI_Datatype datatype = KokkosComm::Impl::mpi_datatype<T>();
    MPI_Put(origin_addr, count, datatype, target_rank, target_disp, count, datatype, win);
  }

  template <typename T>
  void get(T* origin_addr, int count, int source_rank, MPI_Aint source_disp) {
    MPI_Datatype datatype = KokkosComm::Impl::mpi_datatype<T>();
    MPI_Get(origin_addr, count, datatype, source_rank, source_disp, count, datatype, win);
  }

  template <typename T>
  void accumulate(const T* origin_addr, int count, int target_rank, MPI_Aint target_disp, MPI_Op op) {
    MPI_Datatype datatype = KokkosComm::Impl::mpi_datatype<T>();
    MPI_Accumulate(origin_addr, count, datatype, target_rank, target_disp, count, datatype, op, win);
  }

  // Synchronization Functions
  void fence(int assert = 0) { MPI_Win_fence(assert, win); }

  void lock(LockType type, int rank, int assert = 0) {
    int mpi_lock_type = (type == LockType::Exclusive) ? MPI_LOCK_EXCLUSIVE : MPI_LOCK_SHARED;
    MPI_Win_lock(mpi_lock_type, rank, assert, win);
  }

  void unlock(int rank) { MPI_Win_unlock(rank, win); }

  void lock_all(int assert = 0) { MPI_Win_lock_all(assert, win); }

  void unlock_all() { MPI_Win_unlock_all(win); }

  void flush(int rank) { MPI_Win_flush(rank, win); }

  void flush_all() { MPI_Win_flush_all(win); }

  void flush_local(int rank) { MPI_Win_flush_local(rank, win); }

  void flush_local_all() { MPI_Win_flush_local_all(win); }

  // PSCW:
  void post(MPI_Group post_group, int assert = 0) { MPI_Win_post(post_group, assert, win); }

  void start(MPI_Group start_group, int assert = 0) { MPI_Win_start(start_group, assert, win); }

  void complete() { MPI_Win_complete(win); }

  void wait() { MPI_Win_wait(win); }

 private:
  ViewType v_;
  MPI_Comm comm_;
  MPI_Win win;
};

}  // namespace KokkosComm
