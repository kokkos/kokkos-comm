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

#define KOKKOSCOMM_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#if KOKKOSCOMM_GCC_VERSION >= 11400
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include <mpi.h>
#pragma GCC diagnostic pop
#else
#include <mpi.h>  // Maybe we should have this included in the Impl namespace?
#endif

#include <Kokkos_Core.hpp>

#include "KokkosComm_concepts.hpp"
#include "KokkosComm_comm_mode.hpp"

namespace KokkosComm::Impl {

template <typename Scalar>
MPI_Datatype mpi_type() {
  using T = std::decay_t<Scalar>;

  if constexpr (std::is_same_v<T, std::byte>)
    return MPI_BYTE;

  else if constexpr (std::is_same_v<T, char>)
    return MPI_CHAR;
  else if constexpr (std::is_same_v<T, unsigned char>)
    return MPI_UNSIGNED_CHAR;

  else if constexpr (std::is_same_v<T, short>)
    return MPI_SHORT;
  else if constexpr (std::is_same_v<T, unsigned short>)
    return MPI_UNSIGNED_SHORT;

  else if constexpr (std::is_same_v<T, int>)
    return MPI_INT;
  else if constexpr (std::is_same_v<T, unsigned>)
    return MPI_UNSIGNED;

  else if constexpr (std::is_same_v<T, long>)
    return MPI_LONG;
  else if constexpr (std::is_same_v<T, unsigned long>)
    return MPI_UNSIGNED_LONG;

  else if constexpr (std::is_same_v<T, long long>)
    return MPI_LONG_LONG;
  else if constexpr (std::is_same_v<T, unsigned long long>)
    return MPI_UNSIGNED_LONG_LONG;

  else if constexpr (std::is_same_v<T, std::int8_t>)
    return MPI_INT8_T;
  else if constexpr (std::is_same_v<T, std::uint8_t>)
    return MPI_UINT8_T;

  else if constexpr (std::is_same_v<T, std::int16_t>)
    return MPI_INT16_T;
  else if constexpr (std::is_same_v<T, std::uint16_t>)
    return MPI_UINT16_T;

  else if constexpr (std::is_same_v<T, std::int32_t>)
    return MPI_INT32_T;
  else if constexpr (std::is_same_v<T, std::uint32_t>)
    return MPI_UINT32_T;

  else if constexpr (std::is_same_v<T, std::int64_t>)
    return MPI_INT64_T;
  else if constexpr (std::is_same_v<T, std::uint64_t>)
    return MPI_UINT64_T;

  else if constexpr (std::is_same_v<T, std::size_t>) {
    if constexpr (sizeof(std::size_t) == 1) return MPI_UINT8_T;
    if constexpr (sizeof(std::size_t) == 2) return MPI_UINT16_T;
    if constexpr (sizeof(std::size_t) == 4) return MPI_UINT32_T;
    if constexpr (sizeof(std::size_t) == 8) return MPI_UINT64_T;
  }

  else if constexpr (std::is_same_v<T, std::ptrdiff_t>) {
    if constexpr (sizeof(std::ptrdiff_t) == 1) return MPI_INT8_T;
    if constexpr (sizeof(std::ptrdiff_t) == 2) return MPI_INT16_T;
    if constexpr (sizeof(std::ptrdiff_t) == 4) return MPI_INT32_T;
    if constexpr (sizeof(std::ptrdiff_t) == 8) return MPI_INT64_T;
  }

  else if constexpr (std::is_same_v<T, float>)
    return MPI_FLOAT;
  else if constexpr (std::is_same_v<T, double>)
    return MPI_DOUBLE;
  else if constexpr (std::is_same_v<T, long double>)
    return MPI_LONG_DOUBLE;

  else if constexpr (std::is_same_v<T, Kokkos::complex<float>>)
    return MPI_COMPLEX;
  else if constexpr (std::is_same_v<T, Kokkos::complex<double>>)
    return MPI_DOUBLE_COMPLEX;

  else {
    static_assert(std::is_void_v<T>, "mpi_type not implemented");
    return MPI_CHAR;  // unreachable
  }
}

template <typename Scalar>
inline MPI_Datatype mpi_type_v = mpi_type<Scalar>();

class Request {
 private:
  MPI_Request _raw_req;

 public:
  Request(MPI_Request request = MPI_REQUEST_NULL) : _raw_req{request} {}
  operator MPI_Request() const { return _raw_req; }

  void wait() { MPI_Wait(&_raw_req, MPI_STATUS_IGNORE); }
  void free() { MPI_Request_free(&_raw_req); }
  int test() {
    int flag;
    MPI_Test(&_raw_req, &flag, MPI_STATUS_IGNORE);
    return flag;
  }
};

class Reducer {
 private:
  MPI_Op _op;

 public:
  Reducer(MPI_Op op) : _op{op} {}
  operator MPI_Op() const { return _op; }
};

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
  Request isend(SendView send_view, int dest_rank, int tag = 0) const {
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

    return Request{req};
  }

  template <CommMode mode = CommMode::Default, KokkosView RecvView>
  Request irecv(RecvView recv_view, int src_rank, int tag = 0) const {
    KOKKOS_ASSERT(recv_view.span_is_contiguous());
    using T = typename RecvView::value_type;
    MPI_Request req;
    MPI_Irecv(recv_view.data(), recv_view.size(), Impl::mpi_type<T>(), src_rank, tag, _raw_comm, &req);
    return Request{req};
  }

  template <CommMode mode = CommMode::Default, KokkosView SendView, KokkosView RecvView>
  Request isendrecv(SendView send_view, RecvView recv_view, int rank, int tag = 0) const {
    KOKKOS_ASSERT(send_view.span_is_contiguous());
    KOKKOS_ASSERT(recv_view.span_is_contiguous());
    using T = typename SendView::value_type;
    MPI_Request req;
    MPI_Isendrecv(send_view.data(), send_view.size(), Impl::mpi_type<T>(), rank, tag,  //
                  recv_view.data(), recv_view.size(), Impl::mpi_type<T>(), rank, tag, _raw_comm, &req);
    return Request{req};
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
  Request ibarrier() const {
    MPI_Request req;
    MPI_Ibarrier(_raw_comm, &req);
    return Request{req};
  }

  template <CommMode mode = CommMode::Default, KokkosView SendView, KokkosView RecvView>
  Request ireduce(SendView send_view, RecvView recv_view, Reducer op, int root) const {
    static_assert(std::is_same_v<typename SendView::value_type, typename RecvView::value_type>);
    using T = typename SendView::value_type;
    KOKKOS_ASSERT(send_view.span_is_contiguous());
    KOKKOS_ASSERT(recv_view.span_is_contiguous());
    MPI_Request req;
    MPI_Ireduce(send_view.data(), recv_view.data(), send_view.size(), Impl::mpi_type<T>(), op, root, _raw_comm, &req);
    return Request{req};
  }

  template <CommMode mode = CommMode::Default, KokkosView SendView, KokkosView RecvView>
  Request iallreduce(SendView send_view, RecvView recv_view, Reducer op) const {
    static_assert(std::is_same_v<typename SendView::value_type, typename RecvView::value_type>);
    using T = typename SendView::value_type;
    KOKKOS_ASSERT(send_view.span_is_contiguous());
    KOKKOS_ASSERT(recv_view.span_is_contiguous());
    MPI_Request req;
    MPI_Iallreduce(send_view.data(), recv_view.data(), send_view.size(), Impl::mpi_type<T>(), op, _raw_comm, &req);
    return Request{req};
  }
};

}  // namespace KokkosComm::Impl