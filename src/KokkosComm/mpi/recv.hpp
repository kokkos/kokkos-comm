// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>

#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm::mpi {

template <MutKokkosView RecvView>
void recv(const RecvView &rv, int src, int tag, MPI_Comm comm, MPI_Status *status) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::recv");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(rv), "only contiguous views supported for low-level recv");

  using ScalarType = typename RecvView::non_const_value_type;
  MPI_Recv(KokkosComm::data_handle(rv), KokkosComm::span(rv), datatype<MpiSpace, ScalarType>(), src, tag, comm, status);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, MutKokkosView RecvView>
void recv(const ExecSpace &space, RecvView &rv, int src, int tag, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::recv");

  using Packer = typename Impl::PackTraits<RecvView>::packer_type;

  if (!KokkosComm::is_contiguous(rv)) {
    auto args = Packer::allocate_packed_for(space, "packed", rv);
    space.fence("Fence after allocation before MPI_Recv");
    MPI_Recv(KokkosComm::data_handle(args.view), args.count, args.datatype, src, tag, comm, MPI_STATUS_IGNORE);
    Packer::unpack_into(space, rv, args.view);
  } else {
    using RecvScalar = typename RecvView::value_type;
    space.fence("Fence before MPI_Recv");  // prevent work in `space` from writing to recv buffer
    MPI_Recv(
        KokkosComm::data_handle(rv), KokkosComm::span(rv), datatype<MpiSpace, RecvScalar>(), src, tag, comm,
        MPI_STATUS_IGNORE
    );
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::mpi
