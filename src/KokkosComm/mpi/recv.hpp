// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>

#include <KokkosComm/impl/host_staging.hpp>
#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm::mpi {

template <KokkosView RecvView>
void recv(const RecvView &rv, int src, int tag, MPI_Comm comm, MPI_Status *status) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::recv");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(rv), "only contiguous views supported for low-level recv");

  using ScalarType = typename RecvView::non_const_value_type;
  MPI_Recv(KokkosComm::data_handle(rv), KokkosComm::span(rv), datatype<MpiSpace, ScalarType>(), src, tag, comm, status);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void recv(const ExecSpace &space, RecvView &rv, int src, int tag, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::recv");
  using T      = typename RecvView::non_const_value_type;
  using Packer = typename PackTraits<RecvView>::packer_type;

#if defined(KOKKOSCOMM_ENABLE_GPU_AWARE_MPI)
  if (is_contiguous(rv)) {
    space.fence("fence before GPU-aware `MPI_Recv`");  // prevent work in `space` from writing to recv buffer
    MPI_Recv(data_handle(rv), span(rv), datatype<MpiSpace, T>(), src, tag, comm, MPI_STATUS_IGNORE);
  } else {
    auto args = Packer::allocate_packed_for(space, "packed `MPI_Recv`", rv);
    space.fence("fence packing before GPU-aware `MPI_Recv`");
    MPI_Recv(data_handle(args.view), args.count, args.datatype, src, tag, comm, MPI_STATUS_IGNORE);
    Packer::unpack_into(space, rv, args.view);
  }
#else
  auto host_rv = KokkosComm::Impl::stage_for(rv);
  space.fence("fence host staging before `MPI_Recv`");
  if (is_contiguous(host_rv)) {
    MPI_Recv(data_handle(host_rv), span(host_rv), datatype<MpiSpace, T>(), src, tag, comm, MPI_STATUS_IGNORE);
  } else {
    auto args = Packer::allocate_packed_for(space, "packed `MPI_Recv`", host_rv);
    space.fence("fence packing before `MPI_Recv`");
    MPI_Recv(data_handle(args.view), args.count, args.datatype, src, tag, comm, MPI_STATUS_IGNORE);
    // TODO: Can we unpack directly into `rv` instead of `host_rv`?
    Packer::unpack_into(space, host_rv, args.view);
    KokkosComm::Impl::copy_back(space, rv, host_rv);
    space.fence("fence copy back after `MPI_Recv`");
  }
#endif

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::mpi
