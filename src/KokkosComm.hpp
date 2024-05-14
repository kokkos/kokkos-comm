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

#include <Kokkos_Core.hpp>

#include "KokkosComm_base.hpp"
#include "KokkosComm_comm_mode.hpp"
#include "KokkosComm_version.hpp"
#include "KokkosComm_isend.hpp"
#include "KokkosComm_recv.hpp"
#include "KokkosComm_send.hpp"
#include "KokkosComm_concepts.hpp"
#include "KokkosComm_reduce.hpp"
#include "KokkosComm_barrier.hpp"

namespace KokkosComm {

using Impl::Communicator;
using Impl::Reducer;
using Impl::Request;

inline int size(Communicator comm) { return comm.size(); }
inline int rank(Communicator comm) { return comm.rank(); }

template <CommMode mode = CommMode::Default>
inline void send(KokkosView auto send_view, int dest_rank, int tag, Communicator comm) {
  comm.send<mode>(send_view, dest_rank, tag);
}
template <CommMode SendMode = CommMode::Default, KokkosExecutionSpace ExecSpace, KokkosView SendView>
inline void send(const ExecSpace &space, const SendView &sv, int dest, int tag, Communicator comm) {
  return Impl::send<SendMode>(space, sv, dest, tag, comm);
}

template <CommMode mode = CommMode::Default>
inline void recv(KokkosView auto recv_view, int src_rank, int tag, Communicator comm) {
  comm.recv<mode>(recv_view, src_rank, tag);
}
template <CommMode mode = CommMode::Default>
inline void recv(KokkosExecutionSpace auto const &space, KokkosView auto const &sv, int src, int tag,
                 Communicator comm) {
  return Impl::recv(space, sv, src, tag, comm);
}

template <CommMode mode = CommMode::Default>
inline void sendrecv(KokkosView auto send_view, KokkosView auto recv_view, int rank, int tag, Communicator comm) {
  comm.sendrecv<mode>(send_view, recv_view, rank, tag);
}

template <CommMode mode = CommMode::Default>
inline Request isend(KokkosView auto send_view, int dest_rank, int tag, Communicator comm) {
  return comm.isend<mode>(send_view, dest_rank, tag);
}
template <CommMode SendMode = CommMode::Default>
inline Req isend(KokkosExecutionSpace auto const &space, KokkosView auto const &sv, int dest, int tag,
                 Communicator comm) {
  return Impl::isend<SendMode>(space, sv, dest, tag, comm);
}

template <CommMode mode = CommMode::Default>
inline Request irecv(KokkosView auto recv_view, int src_rank, int tag, Communicator comm) {
  return comm.irecv<mode>(recv_view, src_rank, tag);
}

template <CommMode mode = CommMode::Default>
inline Request isendrecv(KokkosView auto send_view, KokkosView auto recv_view, int rank, int tag, Communicator comm) {
  return comm.isendrecv<mode>(send_view, recv_view, rank, tag);
}

inline void barrier(Communicator comm) { comm.barrier(); }

template <CommMode mode = CommMode::Default>
inline void reduce(KokkosView auto send_view, KokkosView auto recv_view, Reducer op, int root, Communicator comm) {
  comm.reduce<mode>(send_view, recv_view, op, root);
}
template <CommMode mode = CommMode::Default>
inline void reduce(KokkosExecutionSpace auto const &space, KokkosView auto const &sv, KokkosView auto const &rv,
                   Reducer op, int root, Communicator comm) {
  return Impl::reduce(space, sv, rv, op, root, comm);
}

template <CommMode mode = CommMode::Default>
inline void allreduce(KokkosView auto send_view, KokkosView auto recv_view, Reducer op, Communicator comm) {
  comm.allreduce<mode>(send_view, recv_view, op);
}

inline Request ibarrier(Communicator comm) { return comm.ibarrier(); }

template <CommMode mode = CommMode::Default>
inline Request ireduce(KokkosView auto send_view, KokkosView auto recv_view, Reducer op, int root, Communicator comm) {
  return comm.ireduce<mode>(send_view, recv_view, op, root);
}

template <CommMode mode = CommMode::Default>
inline Request iallreduce(KokkosView auto send_view, KokkosView auto recv_view, Reducer op, Communicator comm) {
  return comm.iallreduce<mode>(send_view, recv_view, op);
}

}  // namespace KokkosComm
