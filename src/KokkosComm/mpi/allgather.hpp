// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include "mpi_space.hpp"
#include "req.hpp"

#include "impl/types.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm::mpi {

template <KokkosExecutionSpace ExecSpace, KokkosView SView, KokkosView RView>
auto iallgather(const ExecSpace &space, const SView sv, RView rv, MPI_Comm comm) -> Req<MpiSpace> {
  using ST = typename SView::non_const_value_type;
  using RT = typename RView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::mpi::iallgather: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::mpi::iallgather");

  fail_if(!is_contiguous(sv) || !is_contiguous(rv),
          "KokkosComm::mpi::iallgather: unimplemented for non-contiguous views");

  // Sync: Work in space may have been used to produce view data.
  space.fence("fence before non-blocking all-gather");

  Req<MpiSpace> req;
  // All ranks send/recv same count
  MPI_Iallgather(data_handle(sv), span(sv), Impl::mpi_type_v<ST>, data_handle(rv), span(sv), Impl::mpi_type_v<RT>, comm,
                 &req.mpi_request());
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

template <KokkosView SendView, KokkosView RecvView>
void allgather(const SendView &sv, const RecvView &rv, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Mpi::allgather");

  using SendScalar = typename SendView::value_type;
  using RecvScalar = typename RecvView::value_type;

  static_assert(KokkosComm::rank<SendView>() <= 1, "allgather for SendView::rank > 1 not supported");
  static_assert(KokkosComm::rank<RecvView>() <= 1, "allgather for RecvView::rank > 1 not supported");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(sv), "low-level allgather requires contiguous send view");
  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(rv), "low-level allgather requires contiguous recv view");

  const int count = KokkosComm::span(sv);  // all ranks send/recv same count
  MPI_Allgather(KokkosComm::data_handle(sv), count, KokkosComm::Impl::mpi_type_v<SendScalar>,
                KokkosComm::data_handle(rv), count, KokkosComm::Impl::mpi_type_v<RecvScalar>, comm);

  Kokkos::Tools::popRegion();
}

// in-place allgather
template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void allgather(const ExecSpace &space, const RecvView &rv, const size_t recvCount, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Mpi::allgather");

  using RecvScalar = typename RecvView::value_type;

  static_assert(KokkosComm::rank<RecvView>() <= 1, "allgather for RecvView::rank > 1 not supported");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(rv), "low-level allgather requires contiguous recv view");

  space.fence("fence before allgather");  // work in space may have been used to produce send view data
  MPI_Allgather(MPI_IN_PLACE, 0 /*ignored*/, MPI_DATATYPE_NULL /*ignored*/, KokkosComm::data_handle(rv), recvCount,
                KokkosComm::Impl::mpi_type_v<RecvScalar>, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void allgather(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Mpi::allgather");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(sv) || !KokkosComm::is_contiguous(rv),
                           "allgather for non-contiguous views not implemented");

  space.fence("fence before allgather");  // work in space may have been used to produce send view data
  allgather(sv, rv, comm);

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::mpi
