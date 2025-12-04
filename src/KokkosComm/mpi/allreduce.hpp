// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/reduction_op.hpp>
#include "mpi_space.hpp"
#include "req.hpp"

#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace mpi {

template <KokkosView SView, KokkosView RView, KokkosExecutionSpace ExecSpace>
auto iallreduce(const ExecSpace &space, const SView sv, RView rv, MPI_Op op, MPI_Comm comm) -> Req<MpiSpace> {
  using ST = typename SView::non_const_value_type;
  using RT = typename RView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::mpi::iallreduce: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::mpi::iallreduce");

  fail_if(!is_contiguous(sv) || !is_contiguous(rv),
          "KokkosComm::mpi::iallreduce: unimplemented for non-contiguous views");

  // Sync: Work in space may have been used to produce view data.
  space.fence("fence before non-blocking all-gather");

  Req<MpiSpace> req;
  // All ranks send/recv same count
  MPI_Iallreduce(data_handle(sv), data_handle(rv), span(sv), datatype<MpiSpace, ST>(), op, comm, &req.mpi_request());
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

template <KokkosView SendView, KokkosView RecvView>
void allreduce(SendView const &sv, RecvView const &rv, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::allreduce");

  using SendScalar = typename SendView::value_type;
  using RecvScalar = typename RecvView::value_type;
  static_assert(std::is_same_v<std::remove_cv_t<SendScalar>, std::remove_cv_t<RecvScalar> >,
                "Send and receive views have different value types");

  static_assert(KokkosComm::rank<SendView>() <= 1, "allreduce for SendView::rank > 1 not supported");
  static_assert(KokkosComm::rank<RecvView>() <= 1, "allreduce for RecvView::rank > 1 not supported");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(sv), "low-level allreduce requires contiguous send view");
  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(rv), "low-level allreduce requires contiguous recv view");
  KokkosComm::mpi::fail_if(sv.size() != rv.size(), "allreduce requires send and receive views to have the same size");

  int const count = sv.size();
  MPI_Allreduce(KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), count, datatype<MpiSpace, SendScalar>(), op,
                comm);

  Kokkos::Tools::popRegion();
}

template <KokkosView View>
void allreduce(View const &v, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::allreduce");

  using Scalar = typename View::value_type;

  static_assert(KokkosComm::rank<View>() <= 1, "allreduce for View::rank > 1 not supported");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(v), "low-level allgather requires contiguous recv view");

  int const count = v.size();
  MPI_Allreduce(MPI_IN_PLACE, KokkosComm::data_handle(v), count, datatype<MpiSpace, Scalar>(), op, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void allreduce(ExecSpace const &space, SendView const &sv, RecvView const &rv, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::allreduce");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(sv) || !KokkosComm::is_contiguous(rv),
                           "allreduce for non-contiguous views not implemented");

  space.fence("fence before allreduce");  // work in space may have been used to produce send view data
  allreduce(sv, rv, op, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView View>
void allreduce(ExecSpace const &space, View const &v, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::allreduce");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(v), "allreduce for non-contiguous views not implemented");

  space.fence("fence before allreduce");  // work in space may have been used to produce send view data
  allreduce(v, op, comm);

  Kokkos::Tools::popRegion();
}

}  // namespace mpi
namespace Experimental::Impl {

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp, KokkosExecutionSpace ExecSpace>
struct AllReduce<SendView, RecvView, RedOp, ExecSpace, MpiSpace> {
  static auto execute(Handle<ExecSpace, MpiSpace> &h, const SendView &sv, RecvView rv) -> Req<MpiSpace> {
    return mpi::iallreduce(h.space(), sv, rv, reduction_op<MpiSpace, RedOp>(), h.mpi_comm());
  }
};

}  // namespace Experimental::Impl
}  // namespace KokkosComm
