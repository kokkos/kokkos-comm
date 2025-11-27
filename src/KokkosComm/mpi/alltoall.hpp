// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "mpi_space.hpp"
#include "req.hpp"

#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm::mpi {

template <KokkosExecutionSpace ExecSpace, KokkosView SView, KokkosView RView>
auto ialltoall(const ExecSpace &space, const SView sv, RView rv, int count, MPI_Comm comm) -> Req<MpiSpace> {
  using ST = typename SView::non_const_value_type;
  using RT = typename RView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::mpi::ialltoall: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::mpi::ialltoall");

  fail_if(!is_contiguous(sv) || !is_contiguous(rv),
          "KokkosComm::mpi::ialltoall: unimplemented for non-contiguous views");

  // Sync: Work in space may have been used to produce view data.
  space.fence("fence before non-blocking all-gather");

  Req<MpiSpace> req;
  // All ranks send/recv same count
  MPI_Ialltoall(data_handle(sv), count, datatype<MpiSpace, ST>, data_handle(rv), count, datatype<MpiSpace, RT>, comm,
                &req.mpi_request());
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void alltoall(const ExecSpace &space, const SendView &sv, const size_t sendCount, const RecvView &rv,
              const size_t recvCount, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::alltoall");

  using SendScalar = typename SendView::value_type;
  using RecvScalar = typename RecvView::value_type;

  static_assert(KokkosComm::rank<SendView>() <= 1, "alltoall for SendView::rank > 1 not supported");
  static_assert(KokkosComm::rank<RecvView>() <= 1, "alltoall for RecvView::rank > 1 not supported");

  // Make sure views are ready
  space.fence("KokkosComm::mpi::alltoall");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(sv) || !KokkosComm::is_contiguous(rv),
                           "alltoall for non-contiguous views not implemented");

  int size;
  MPI_Comm_size(comm, &size);

  if (sendCount * size > KokkosComm::extent(sv, 0)) {
    std::stringstream ss;
    ss << "alltoall sendCount * communicator size (" << sendCount << " * " << size
       << ") is greater than send view size";
    KokkosComm::mpi::fail_if(true, ss.str().data());
  }
  if (recvCount * size > KokkosComm::extent(rv, 0)) {
    std::stringstream ss;
    ss << "alltoall recvCount * communicator size (" << recvCount << " * " << size
       << ") is greater than recv view size";
    KokkosComm::mpi::fail_if(true, ss.str().data());
  }

  MPI_Alltoall(KokkosComm::data_handle(sv), sendCount, datatype<MpiSpace, SendScalar>(), KokkosComm::data_handle(rv),
               recvCount, datatype<MpiSpace, RecvScalar>(), comm);

  Kokkos::Tools::popRegion();
}

// in-place alltoall
template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void alltoall(const ExecSpace &space, const RecvView &rv, const size_t recvCount, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::alltoall");

  using RecvScalar = typename RecvView::value_type;

  static_assert(RecvView::rank <= 1, "alltoall for RecvView::rank > 1 not supported");

  // Make sure views are ready
  space.fence("KokkosComm::mpi::alltoall");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(rv), "alltoall for non-contiguous views not implemented");

  int size;
  MPI_Comm_size(comm, &size);

  if (recvCount * size > KokkosComm::extent(rv, 0)) {
    std::stringstream ss;
    ss << "alltoall recvCount * communicator size (" << recvCount << " * " << size
       << ") is greater than recv view size";
    KokkosComm::mpi::fail_if(true, ss.str().data());
  }

  MPI_Alltoall(MPI_IN_PLACE, 0 /*ignored*/, MPI_BYTE /*ignored*/, KokkosComm::data_handle(rv), recvCount,
               datatype<MpiSpace, RecvScalar>(), comm);

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::mpi
