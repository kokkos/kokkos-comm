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

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
auto send(const ExecSpace& space, const SendView& sv, int peer, ncclComm_t comm) -> Request<RcclSpace> {
  using T = typename SendView::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::rccl::send");

  Request<RcclSpace> req;
  if (is_contiguous(sv)) {
    KC_RCCL_CHECK(ncclSend(data_handle(sv), span(sv), datatype_for<RcclSpace>(sv), peer, comm, space.hip_stream()));
  } else {
    using Packer = typename Impl::PackTraits<SendView>::packer_type;
    auto pckd_sv = Packer::pack(space, "pckd_sv", sv);
    KC_RCCL_CHECK(
        ncclSend(data_handle(pckd_sv.view_), pckd_sv.count_, pckd_sv.datatype_, peer, comm, space.hip_stream())
    );
    req.extend_view_lifetime(pckd_sv.view_);
  }
  req.capture_stream_state(space.hip_stream());
  req.extend_view_lifetime(sv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace Experimental::rccl
namespace Impl {

template <KokkosView SendView>
struct Send<SendView, Kokkos::HIP, Experimental::RcclSpace> {
  static auto execute(Communicator<Experimental::RcclSpace, Kokkos::HIP>& h, SendView sv, int peer)
      -> Request<Experimental::RcclSpace> {
    return Experimental::rccl::send(h.exec(), sv, peer, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm
