// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "nccl_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace Experimental::nccl {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
auto send(const ExecSpace& space, const SendView& sv, int peer, ncclComm_t comm) -> Request<NcclSpace> {
  using T = typename SendView::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::Impl::send");

  Request<NcclSpace> req;
  if (is_contiguous(sv)) {
    KC_NCCL_CHECK(ncclSend(data_handle(sv), span(sv), datatype<NcclSpace, T>(), peer, comm, space.cuda_stream()));
  } else {
    using Packer = typename Impl::PackTraits<SendView>::packer_type;
    auto pckd_sv = Packer::pack(space, "pckd_sv", sv);
    KC_NCCL_CHECK(
        ncclSend(data_handle(pckd_sv.view_), pckd_sv.count_, pckd_sv.datatype_, peer, comm, space.cuda_stream())
    );
    req.capture_stream_state(space.cuda_stream());
    req.extend_view_lifetime(pckd_sv.view_);
  }
  req.extend_view_lifetime(sv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace Experimental::nccl
namespace Impl {

template <KokkosView SendView>
struct Send<SendView, Kokkos::Cuda, Experimental::NcclSpace> {
  static auto execute(Communicator<Experimental::NcclSpace, Kokkos::Cuda>& h, SendView sv, int peer)
      -> Request<Experimental::NcclSpace> {
    return Experimental::nccl::send(h.exec(), sv, peer, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm
