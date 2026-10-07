// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/error.hpp>
#include "nccl_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace Experimental::nccl {

template <KokkosExecutionSpace ExecSpace, MutKokkosView RecvView>
auto recv(const ExecSpace& space, RecvView& rv, int peer, ncclComm_t comm) -> Request<NcclSpace> {
  using T = typename RecvView::non_const_value_type;
  Kokkos::Profiling::ScopedRegion region("KokkosComm::Impl::recv");

  Request<NcclSpace> req;
  if (is_contiguous(rv)) {
    KC_NCCL_CHECK_REQ(ncclRecv(data_handle(rv), span(rv), datatype<NcclSpace, T>(), peer, comm, space.cuda_stream()));
  } else {
    using Packer = typename Impl::PackTraits<RecvView>::packer_type;
    auto pckd_rv = Packer::allocate_packed_for(space, "pckd_rv", rv);
    KC_NCCL_CHECK_REQ(
        ncclRecv(data_handle(pckd_rv.view_), pckd_rv.count_, pckd_rv.datatype_, peer, comm, space.cuda_stream())
    );
    req.add_callback([space, rv, pckd_rv]() {
      Packer::unpack_into(space, rv, pckd_rv.view_);
      space.fence("fence `pckd_rv` unpacking after NCCL call");
    });
  }
  req.capture_stream_state(space.cuda_stream());
  req.extend_view_lifetime(rv);

  return req;
}

}  // namespace Experimental::nccl
namespace Impl {

template <MutKokkosView RecvView>
struct Recv<RecvView, Kokkos::Cuda, Experimental::NcclSpace> {
  static auto execute(Communicator<Experimental::NcclSpace, Kokkos::Cuda>& h, RecvView sv, int peer)
      -> Request<Experimental::NcclSpace> {
    return Experimental::nccl::recv(h.exec(), sv, peer, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm
