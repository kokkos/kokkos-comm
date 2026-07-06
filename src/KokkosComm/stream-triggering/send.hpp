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

#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm::Experimental {
namespace stream {
  
template <KokkosExecutionSpace ExecSpace, KokkosView SendView, CommunicationMode SendMode>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm, MPI_Info mem_info, MPIS_Request* reqs, SendMode) {
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::stream::send");
  using T      = typename SendView::non_const_value_type;
  using Packer = typename KokkosComm::PackTraits<SendView>::packer_type;

  auto mpi_send_fn = [dest, tag, comm, mem_info, reqs](void *view, int cnt, MPI_Datatype dtype) {
    if constexpr (std::is_same_v<SendMode, CommModeStandard>) {
      MPIS_Send_init(view, cnt, dtype, dest, tag, comm, mem_info, reqs);
    } else if constexpr (std::is_same_v<SendMode, CommModeReady>) {
      //MPI_Rsend(view, cnt, dtype, dest, tag, comm);
      // MPIS_RSend_init(a.data(), a.size(), dtype, dest, tag, comm, mem_info, ctx->reqs);)
static_assert(std::is_void_v<SendMode>, "KokkosComm::Experimental::stream::send: Ready Mode not enable");
    } else if constexpr (std::is_same_v<SendMode, CommModeSynchronous>) {
static_assert(std::is_void_v<SendMode>, "KokkosComm::Experimental::stream::send: Synchronous Mode not enable");
    } else {
      static_assert(std::is_void_v<SendMode>, "KokkosComm::Experimental::stream::send: unexpected communication mode");
    }
  };

  if (is_contiguous(sv)) {
    mpi_send_fn(data_handle(sv), span(sv), datatype<MpiSpace, T>());
  } else {
    //std::cerr << "Send: not contig!" << std::endl;
    auto args = Packer::pack(space, sv);
    mpi_send_fn(data_handle(args.view), args.count, args.datatype);
  }

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm, MPI_Info mem_info, MPIS_Request* reqs) {
  send(space, sv, dest, tag, comm, mem_info, reqs, DefaultCommMode{});
}

/// NOTE: This overload has the side effect of fencing on the default execution space.
template <KokkosView SendView>
void send(const SendView &sv, int dest, int tag, MPI_Comm comm, MPI_Info mem_info, MPIS_Request* reqs) {
  send(Kokkos::DefaultExecutionSpace(), sv, dest, tag, comm, mem_info, reqs, DefaultCommMode{});
}
  
} // namespace stream
}  // namespace KokkosComm
