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

template <KokkosExecutionSpace ExecSpace>
class Universe {
 public:
  static auto create(MPI_Session shandle, MPI_Comm comm) -> Universe { return Universe(shandle, comm); }

  auto free(void) -> void {
    // Detach the buffer from MPI. It's ok to destroy at scope exit as it won't have anything to do with MPI anymore.
    detach_buffer();

    // Eagerly destruct the communicator: we cannot rely on the universe going out of scope as `MPI_Finalize` will be
    // called earlier than that.
    MPI_Comm comm = _comm.as_raw();
    MPI_Comm_free(&comm);

    MPI_Session_finalize(&_shandle);
  }

  auto set_buffer_size(size_t size) -> void {
    detach_buffer();
    if (0 < size) {
      _buffer.resize(size + MPI_BSEND_OVERHEAD);
      MPI_Buffer_attach(_buffer.data(), _buffer.size());
      _is_buffer_attached = true;
    }
  }

  auto detach_buffer(void) -> void {
    if (_is_buffer_attached) {
      int size;
      MPI_Buffer_detach(_buffer.data(), &size);
      assert(static_cast<size_t>(size) == _buffer.size());  // safety check
      _is_buffer_attached = false;
    }
  }

  auto comm(void) -> Communicator<ExecSpace> { return _comm; }

 private:
  Universe(MPI_Session shandle, MPI_Comm comm) : Universe(shandle, comm, 0) {}

  Universe(MPI_Session shandle, MPI_Comm comm, size_t buf_size)
      : _shandle(shandle),
        _comm(Communicator<ExecSpace>::from_raw_unchecked(comm)),
        _buffer(buf_size),
        _is_buffer_attached(false) {}

  MPI_Session _shandle;
  Communicator<ExecSpace> _comm;
  std::vector<uint8_t> _buffer;
  bool _is_buffer_attached;
};

template <KokkosExecutionSpace ExecSpace>
auto initialize(void) -> Universe<ExecSpace> {
  MPI_Info kokkoscomm_info = MPI_INFO_NULL;
  MPI_Info_create(&kokkoscomm_info);

  // Set threading level for our session
  constexpr char thrd_lvl_key[] = "thread_level";
  constexpr char thrd_lvl_val[] = "MPI_THREAD_MULTIPLE";
  MPI_Info_set(kokkoscomm_info, thrd_lvl_key, thrd_lvl_val);
  // TODO: error handling

#ifdef KOKKOSCOMM_CUDA_AWARE_MPI
  // Disable CUDA pointer attribute checks from MPI
  constexpr char cu_ptr_attr_key[] = "mpi_communication_pattern";
  constexpr char cu_ptr_attr_val[] = "MPI_CPU_TO_GPU";
  MPI_Info_set(kokkoscomm_info, cu_ptr_attr_key, cu_ptr_attr_val);
  // TODO: error handling
#endif

  MPI_Session kokkoscomm_shandle = MPI_SESSION_NULL;
  MPI_Session_init(kokkoscomm_info, MPI_ERRORS_RETURN, &kokkoscomm_shandle);
  // TODO: error handling

  MPI_Group kokkoscomm_group = MPI_GROUP_NULL;
  constexpr char pset_name[] = "mpi://WORLD";
  MPI_Group_from_session_pset(kokkoscomm_shandle, pset_name, &kokkoscomm_group);
  // TODO: error handling

  MPI_Comm kokkoscomm_comm = MPI_COMM_NULL;
  MPI_Comm_create_from_group(kokkoscomm_group, "kokkos-comm.default_session", MPI_INFO_NULL, MPI_ERRORS_RETURN,
                             &kokkoscomm_comm);
  // TODO: error handling

  // Resource release
  MPI_Group_free(&kokkoscomm_group);
  MPI_Info_free(&kokkoscomm_info);

  return Universe<ExecSpace>::create(kokkoscomm_shandle, kokkoscomm_comm);
}

template <KokkosExecutionSpace ExecSpace>
auto initialize(int &argc, char *argv[]) -> Universe<ExecSpace> {
  // Check that MPI was initiliazed and init if it wasn't
  int is_initialized;
  MPI_Initialized(&is_initialized);
  if (0 == is_initialized) {
    int required = MPI_THREAD_MULTIPLE, provided;
    MPI_Init_thread(&argc, &argv, required, &provided);
  }

  return initialize<ExecSpace>();
}

template <KokkosExecutionSpace ExecSpace>
auto finalize(Universe<ExecSpace> &universe) -> void {
  universe.free();
}

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
