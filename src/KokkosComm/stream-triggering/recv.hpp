// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <stream-triggering.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "comm_mode.hpp"
#include "context.hpp"

#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm::Experimental {
namespace stream {

template <KokkosView RecvView>
void recv(const RecvView &rv, int src, int tag, MPI_Comm comm, MPI_Info mem_info, MPIS_Request* reqs) {
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::stream::recv");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(rv), "only contiguous views supported for low-level recv");

  using ScalarType = typename RecvView::non_const_value_type;
  MPIS_Recv_init(KokkosComm::data_handle(rv), KokkosComm::span(rv), datatype<MpiSpace, ScalarType>(), src, tag, comm, mem_info, reqs);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void recv(const ExecSpace &space, RecvView &rv, int src, int tag, MPI_Comm comm, MPI_Info mem_info, MPIS_Request* reqs) {
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::stream::recv");

  using KCPT   = KokkosComm::PackTraits<RecvView>;
  using Packer = typename KCPT::packer_type;
  using Args   = typename Packer::args_type;

  if (!KokkosComm::is_contiguous(rv)) {
    Args args = Packer::allocate_packed_for(space, "packed", rv);
    MPIS_Recv_init(KokkosComm::data_handle(args.view), KokkosComm::span(args.view), args.datatype, src, tag, comm, mem_info, reqs);
    space.fence("ensure prints are correct!");
    // packer should not be here! or should wait until over
    // something, either the lifetime of args.view or packer getting info before request is over
    Packer::unpack_into(space, rv, args.view);
    space.fence("ensure prints are correct!");
  } else {
    using RecvScalar = typename RecvView::value_type;
    MPIS_Recv_init(KokkosComm::data_handle(rv), KokkosComm::span(rv), datatype<MpiSpace, RecvScalar>(), src, tag, comm, mem_info, reqs);
  }

  Kokkos::Tools::popRegion();
}

  template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void recv(const ExecSpace &space, RecvView &rv, int src, int tag, MPI_Comm comm, StreamContext& context, MPIS_Request* reqs) {
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::stream::recv");

  using KCPT   = KokkosComm::PackTraits<RecvView>;
  using Packer = typename KCPT::packer_type;
  using Args   = typename Packer::args_type;

  if (!KokkosComm::is_contiguous(rv)) {
    Args args = Packer::allocate_packed_for(space, "packed", rv);
    MPIS_Recv_init(KokkosComm::data_handle(args.view), KokkosComm::span(args.view), args.datatype, src, tag, comm, context.get_mem_info(), reqs);
    space.fence("ensure prints are correct!");
    context.block(1, reqs);
    std::cerr << "blocked!" << std::endl;
    space.fence("ensure prints are correct!");
    Packer::unpack_into(space, rv, args.view);
  } else {
    using RecvScalar = typename RecvView::value_type;
    MPIS_Recv_init(KokkosComm::data_handle(rv), KokkosComm::span(rv), datatype<MpiSpace, RecvScalar>(), src, tag, comm, context.get_mem_info(), reqs);
  }

  Kokkos::Tools::popRegion();
}
  
}  // namespace stream
}  // namespace KokkosComm
