// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "nccl_space.hpp"

#include <KokkosComm/impl/contiguous.hpp>
#include "impl/pack_traits.hpp"
#include "impl/nccl_check.hpp"

namespace KokkosComm {
namespace Experimental::nccl {

namespace KC = KokkosComm;

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
auto recv(const ExecSpace &space, RecvView &rv, int peer, ncclComm_t comm) -> Req<NcclSpace> {
  using T = typename RecvView::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::Impl::recv");

  Req<NcclSpace> req{space.cuda_stream()};
  if (KC::is_contiguous(rv)) {
    KC_NCCL_CHECK(
        ncclRecv(KC::data_handle(rv), KC::span(rv), datatype<NcclSpace, T>(), peer, comm, space.cuda_stream()));
  } else {
    using Packer = typename Impl::PackTraits<RecvView>::packer_type;
    auto pckd_rv = KC::Impl::allocate_contiguous_for(space, "KC::nccl::recv pckd_rv", rv);
    KC_NCCL_CHECK(ncclRecv(KC::data_handle(pckd_rv), KC::span(pckd_rv), datatype<NcclSpace, T>(), peer, comm,
                           space.cuda_stream()));
    Packer::unpack_into(space, rv, pckd_rv);
    req.extend_view_lifetime(pckd_rv);
  }
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace Experimental::nccl
namespace Impl {

template <KokkosView RecvView>
struct Recv<RecvView, Kokkos::Cuda, Experimental::NcclSpace> {
  static auto execute(Handle<Kokkos::Cuda, Experimental::NcclSpace> &h, RecvView sv, int peer)
      -> Req<Experimental::NcclSpace> {
    return Experimental::nccl::recv(h.space(), sv, peer, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm
