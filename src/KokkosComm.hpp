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

#include <mpi.h>
#include "KokkosComm_collective.hpp"
#include "KokkosComm_version.hpp"
#include "KokkosComm_isend.hpp"
#include "KokkosComm_recv.hpp"
#include "KokkosComm_send.hpp"
#include "KokkosComm_concepts.hpp"
#include "KokkosComm_comm_mode.hpp"
#include "KokkosComm_communicator.hpp"

#include <Kokkos_Core.hpp>

namespace KokkosComm {
auto initialize(int &argc, char *argv[]) -> Universe {}

class Universe {
 public:
  ~Universe() {
    detach_buffer();

    MPI_Session_finalize(&_shandle);

    int is_finalized;
    MPI_Finalized(&is_finalized);
    if (0 == is_finalized) {
      MPI_Finalize();
    }
  }

  auto set_buffer_size(size_t size) -> void {
    detach_buffer();
    if (0 < size) {
      _buffer.resize(size);
      MPI_Buffer_attach(_buffer.data(), _buffer.size());
    }
  }

  auto detach_buffer(void) -> void {
    int size;
    MPI_Buffer_detach(_buffer.data(), &size);
    assert(static_cast<size_t>(size) == _buffer.size());  // safety check
  }

 private:
  enum class Threading {
    Single = MPI_THREAD_SINGLE,
    Funneled = MPI_THREAD_FUNNELED,
    Serialized = MPI_THREAD_SERIALIZED,
    Multiple = MPI_THREAD_MULTIPLE,
  };

  Universe(int &argc, char *argv[]) : Universe(argc, argv, 0) {}

  Universe(int &argc, char *argv[], size_t buf_size) : _buffer(buf_size) {
    int is_initialized, provided;

    MPI_Initialized(&is_initialized);
    if (0 == is_initialized) {
      int required = MPI_THREAD_MULTIPLE;
      MPI_Init_thread(&argc, &argv, required, &provided);
    } else {
      MPI_Query_thread(&provided);
    }
    _threading = static_cast<Threading>(provided);

    const char comm_key[] = "mpi_communication_pattern";
    const char comm_val[] = "MPI_CPU_TO_GPU";
    MPI_Info lib_info = MPI_INFO_NULL;
    MPI_Info_create(&lib_info);
    MPI_Info_set(lib_info, comm_key, comm_val);

    MPI_Session lib_handle = MPI_SESSION_NULL;
    MPI_Session_init(lib_info, MPI_ERRORS_RETURN, &lib_handle);
    _shandle = lib_handle;
  }

  MPI_Session _shandle;
  Threading _threading;
  std::vector<uint8_t> _buffer;
};

template <CommMode SendMode = CommMode::Default, KokkosExecutionSpace ExecSpace, KokkosView SendView>
Req isend(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  return Impl::isend<SendMode>(space, sv, dest, tag, comm);
}

template <CommMode SendMode = CommMode::Default, KokkosExecutionSpace ExecSpace, KokkosView SendView>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  return Impl::send<SendMode>(space, sv, dest, tag, comm);
}

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void recv(const ExecSpace &space, RecvView &sv, int src, int tag, MPI_Comm comm) {
  return Impl::recv(space, sv, src, tag, comm);
}

}  // namespace KokkosComm
