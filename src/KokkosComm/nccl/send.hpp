// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "nccl_space.hpp"

#include "impl/pack_traits.hpp"
#include "impl/nccl_check.hpp"

namespace KokkosComm {
namespace Experimental::nccl {

namespace KC = KokkosComm;

template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
auto send(const ExecSpace& space, const SendView& sv, int peer, ncclComm_t comm) -> Req<NcclSpace> {
  using T = typename SendView::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::Impl::send");

  Req<NcclSpace> req{space.cuda_stream()};
  if (KC::is_contiguous(sv)) {
    KC_NCCL_CHECK(
        ncclSend(KC::data_handle(sv), KC::span(sv), datatype<NcclSpace, T>(), peer, comm, space.cuda_stream()));
  } else {
    using Packer = typename Impl::PackTraits<SendView>::packer_type;
    auto args    = Packer::pack(space, sv);
    KC_NCCL_CHECK(
        ncclSend(KC::data_handle(args.view_), args.count_, datatype<NcclSpace, T>(), peer, comm, space.cuda_stream()));
    req.extend_view_lifetime(args.view_);
  }
  req.extend_view_lifetime(sv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace Experimental::nccl
namespace Impl {

template <KokkosView SendView>
struct Send<SendView, Kokkos::Cuda, Experimental::NcclSpace> {
  static auto execute(Handle<Kokkos::Cuda, Experimental::NcclSpace>& h, SendView sv, int peer)
      -> Req<Experimental::NcclSpace> {
    return Experimental::nccl::send(h.space(), sv, peer, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm
