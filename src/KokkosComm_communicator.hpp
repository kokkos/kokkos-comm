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
#include "KokkosComm_request.hpp"
#include "KokkosComm_concepts.hpp"
#include "KokkosComm_comm_mode.hpp"
#include "KokkosComm_reducer.hpp"

namespace KokkosComm {

class Communicator {
 private:
  MPI_Comm _raw_comm = MPI_COMM_WORLD;

 public:
  Communicator(MPI_Comm mpi_communicator) : _raw_comm{mpi_communicator} {}
  operator MPI_Comm() { return _raw_comm; }

  int size() const {
    int size;
    MPI_Comm_size(_raw_comm, &size);
    return size;
  }

  int rank() const {
    int rank;
    MPI_Comm_rank(_raw_comm, &rank);
    return rank;
  }

  // Blocking point to point
  template <CommMode mode = CommMode::Default, KokkosView SendView>
  void send(SendView send_view, int dest_rank, int tag = 0) const {
    KOKKOS_ASSERT(send_view.span_is_contiguous());
    using T = typename SendView::value_type;

    if constexpr (mode == CommMode::Standard)
      MPI_Send(send_view.data(), send_view.size(), Impl::mpi_type<T>(), dest_rank, tag, _raw_comm);
    else if constexpr (mode == CommMode::Ready)
      MPI_Rsend(send_view.data(), send_view.size(), Impl::mpi_type<T>(), dest_rank, tag, _raw_comm);
    else if constexpr (mode == CommMode::Synchronous)
      MPI_Ssend(send_view.data(), send_view.size(), Impl::mpi_type<T>(), dest_rank, tag, _raw_comm);
    else if constexpr (mode == CommMode::Default)
#ifdef KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE
      MPI_Ssend(send_view.data(), send_view.size(), Impl::mpi_type<T>(), dest_rank, tag, _raw_comm);
#else
      MPI_Send(send_view.data(), send_view.size(), Impl::mpi_type<T>(), dest_rank, tag, _raw_comm);
#endif
    else
      static_assert(std::is_void_v<SendView>, "Unknown communication mode");
  }

  template <CommMode mode = CommMode::Default, KokkosView RecvView>
  void recv(RecvView recv_view, int src_rank, int tag = 0) const {
    KOKKOS_ASSERT(recv_view.span_is_contiguous());
    using T = typename RecvView::value_type;
    MPI_Recv(recv_view.data(), recv_view.size(), Impl::mpi_type<T>(), src_rank, tag, _raw_comm, MPI_STATUS_IGNORE);
  }

  template <CommMode mode = CommMode::Default, KokkosView SendView, KokkosView RecvView>
  void sendrecv(SendView send_view, RecvView recv_view, int rank, int tag = 0) const {
    KOKKOS_ASSERT(send_view.span_is_contiguous());
    KOKKOS_ASSERT(recv_view.span_is_contiguous());
    using T = typename SendView::value_type;
    MPI_Sendrecv(send_view.data(), send_view.size(), Impl::mpi_type<T>(), rank, tag,  //
                 recv_view.data(), recv_view.size(), Impl::mpi_type<T>(), rank, tag, _raw_comm, MPI_STATUS_IGNORE);
  }

  // Async point to point
  template <CommMode mode = CommMode::Default, KokkosView SendView>
  KokkosComm::Request isend(SendView send_view, int dest_rank, int tag = 0) const {
    KOKKOS_ASSERT(send_view.span_is_contiguous());
    using T = typename SendView::value_type;
    MPI_Request req;

    if constexpr (mode == CommMode::Standard)
      MPI_Isend(send_view.data(), send_view.size(), Impl::mpi_type<T>(), dest_rank, tag, _raw_comm, &req);
    else if constexpr (mode == CommMode::Ready)
      MPI_Irsend(send_view.data(), send_view.size(), Impl::mpi_type<T>(), dest_rank, tag, _raw_comm, &req);
    else if constexpr (mode == CommMode::Synchronous)
      MPI_Issend(send_view.data(), send_view.size(), Impl::mpi_type<T>(), dest_rank, tag, _raw_comm, &req);
    else if constexpr (mode == CommMode::Default)
#ifdef KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE
      MPI_Issend(send_view.data(), send_view.size(), Impl::mpi_type<T>(), dest_rank, tag, _raw_comm, &req);
#else
      MPI_Isend(send_view.data(), send_view.size(), Impl::mpi_type<T>(), dest_rank, tag, _raw_comm, &req);
#endif
    else
      static_assert(std::is_void_v<SendView>, "Unknown communication mode");

    return KokkosComm::Request{req};
  }

  template <CommMode mode = CommMode::Default, KokkosView RecvView>
  KokkosComm::Request irecv(RecvView recv_view, int src_rank, int tag = 0) const {
    KOKKOS_ASSERT(recv_view.span_is_contiguous());
    using T = typename RecvView::value_type;
    MPI_Request req;
    MPI_Irecv(recv_view.data(), recv_view.size(), Impl::mpi_type<T>(), src_rank, tag, _raw_comm, &req);
    return KokkosComm::Request{req};
  }

  template <CommMode mode = CommMode::Default, KokkosView SendView, KokkosView RecvView>
  KokkosComm::Request isendrecv(SendView send_view, RecvView recv_view, int rank, int tag = 0) const {
    KOKKOS_ASSERT(send_view.span_is_contiguous());
    KOKKOS_ASSERT(recv_view.span_is_contiguous());
    using T = typename SendView::value_type;
    MPI_Request req;
    MPI_Isendrecv(send_view.data(), send_view.size(), Impl::mpi_type<T>(), rank, tag,  //
                  recv_view.data(), recv_view.size(), Impl::mpi_type<T>(), rank, tag, _raw_comm, &req);
    return KokkosComm::Request{req};
  }

  // Blocking collective
  void barrier() const { MPI_Barrier(_raw_comm); }

  template <CommMode mode = CommMode::Default, KokkosView SendView, KokkosView RecvView>
  void reduce(SendView send_view, RecvView recv_view, Reducer op, int root) const {
    static_assert(std::is_same_v<typename SendView::value_type, typename RecvView::value_type>);
    using T = typename SendView::value_type;
    KOKKOS_ASSERT(send_view.span_is_contiguous());
    KOKKOS_ASSERT(recv_view.span_is_contiguous());
    MPI_Reduce(send_view.data(), recv_view.data(), send_view.size(), Impl::mpi_type<T>(), op, root, _raw_comm);
  }

  template <CommMode mode = CommMode::Default, KokkosView SendView, KokkosView RecvView>
  void allreduce(SendView send_view, RecvView recv_view, Reducer op) const {
    static_assert(std::is_same_v<typename SendView::value_type, typename RecvView::value_type>);
    using T = typename SendView::value_type;
    KOKKOS_ASSERT(send_view.span_is_contiguous());
    KOKKOS_ASSERT(recv_view.span_is_contiguous());
    MPI_Allreduce(send_view.data(), recv_view.data(), send_view.size(), Impl::mpi_type<T>(), op, _raw_comm);
  }

  // Async collective
  KokkosComm::Request ibarrier() const {
    MPI_Request req;
    MPI_Ibarrier(_raw_comm, &req);
    return KokkosComm::Request{req};
  }
};

// Free function equivalents
inline int size(Communicator comm) { return comm.size(); }
inline int rank(Communicator comm) { return comm.rank(); }

template <CommMode mode = CommMode::Default>
inline void send(KokkosView auto send_view, int dest_rank, int tag, Communicator comm) {
  comm.send<mode>(send_view, dest_rank, tag);
}
template <CommMode mode = CommMode::Default>
inline void recv(KokkosView auto recv_view, int src_rank, int tag, Communicator comm) {
  comm.recv<mode>(recv_view, src_rank, tag);
}
template <CommMode mode = CommMode::Default>
inline void sendrecv(KokkosView auto send_view, KokkosView auto recv_view, int rank, int tag, Communicator comm) {
  comm.sendrecv<mode>(send_view, recv_view, rank, tag);
}

template <CommMode mode = CommMode::Default>
inline Request isend(KokkosView auto send_view, int dest_rank, int tag, Communicator comm) {
  return comm.isend<mode>(send_view, dest_rank, tag);
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
inline void allreduce(KokkosView auto send_view, KokkosView auto recv_view, Reducer op, Communicator comm) {
  comm.allreduce<mode>(send_view, recv_view, op);
}

inline Request ibarrier(Communicator comm) { return comm.ibarrier(); }

}  // namespace KokkosComm