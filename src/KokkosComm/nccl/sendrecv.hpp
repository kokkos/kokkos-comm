// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <nccl.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/fwd.hpp>
#include "nccl_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace Experimental::nccl {

/// @brief Enqueues a send and receive together on the execution space's CUDA stream.
template <KokkosView SendV, MutKokkosView RecvV>
auto sendrecv(Kokkos::Cuda const& exec, SendV const& sv, int dst, RecvV const& rv, int src, ncclComm_t comm)
    -> Request<NcclSpace> {
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::sendrecv");

  Request<NcclSpace> req;

  const void* sbuf = data_handle(sv);
  auto scnt        = span(sv);
  auto sdtype      = datatype_for<NcclSpace>(sv);
  if (!is_contiguous(sv)) {
    using SendPckr = typename Impl::PackTraits<SendV>::packer_type;
    auto pckd_sv   = SendPckr::pack(exec, "pckd_sv", sv);
    sbuf           = data_handle(pckd_sv.view_);
    scnt           = pckd_sv.count_;
    sdtype         = pckd_sv.datatype_;
    req.extend_view_lifetime(pckd_sv.view_);
  }

  void* rbuf  = data_handle(rv);
  auto rcnt   = span(rv);
  auto rdtype = datatype_for<NcclSpace>(rv);
  if (!is_contiguous(rv)) {
    using RecvPckr = typename Impl::PackTraits<RecvV>::packer_type;
    auto pckd_rv   = RecvPckr::allocate_packed_for(exec, "pckd_rv", rv);
    rbuf           = data_handle(pckd_rv.view_);
    rcnt           = pckd_rv.count_;
    rdtype         = pckd_rv.datatype_;
    // Capturing both views keeps them alive until the receive is unpacked.
    req.add_callback([exec, rv, pckd_rv]() {
      RecvPckr::unpack_into(exec, rv, pckd_rv.view_);
      exec.fence("fence `pckd_rv` unpacking after NCCL call");
    });
  }

  KC_NCCL_CHECK(ncclGroupStart());
  KC_NCCL_CHECK(ncclSend(sbuf, scnt, sdtype, dst, comm, exec.cuda_stream()));
  KC_NCCL_CHECK(ncclRecv(rbuf, rcnt, rdtype, src, comm, exec.cuda_stream()));
  KC_NCCL_CHECK(ncclGroupEnd());
  // NCCL enqueues the grouped operations when the group ends.
  req.capture_stream_state(exec.cuda_stream());
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace Experimental::nccl
}  // namespace KokkosComm
