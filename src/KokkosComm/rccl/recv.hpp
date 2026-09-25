// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <rccl/rccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "rccl_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace Experimental::rccl {

template <KokkosExecutionSpace ExecSpace, MutKokkosView RecvView>
auto recv(const ExecSpace& space, RecvView& rv, int peer, ncclComm_t comm) -> Request<RcclSpace> {
  using T = typename RecvView::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::rccl::recv");

  Request<RcclSpace> req;
  if (is_contiguous(rv)) {
    KC_RCCL_CHECK(ncclRecv(data_handle(rv), span(rv), datatype_for<RcclSpace>(rv), peer, comm, space.hip_stream()));
  } else {
    using Packer = typename Impl::PackTraits<RecvView>::packer_type;
    auto pckd_rv = Packer::allocate_packed_for(space, "pckd_rv", rv);
    KC_RCCL_CHECK(
        ncclRecv(data_handle(pckd_rv.view_), pckd_rv.count_, pckd_rv.datatype_, peer, comm, space.hip_stream())
    );
    req.add_callback([space, rv, pckd_rv]() {
      Packer::unpack_into(space, rv, pckd_rv.view_);
      space.fence("fence `pckd_rv` unpacking after RCCL call");
    });
  }
  req.capture_stream_state(space.hip_stream());
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace Experimental::rccl
namespace Impl {

template <MutKokkosView RecvView>
struct Recv<RecvView, Kokkos::HIP, Experimental::RcclSpace> {
  static auto execute(Communicator<Experimental::RcclSpace, Kokkos::HIP>& h, RecvView sv, int peer)
      -> Request<Experimental::RcclSpace> {
    return Experimental::rccl::recv(h.exec(), sv, peer, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm
