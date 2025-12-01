// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>

#include "impl/pack_traits.hpp"
#include "impl/types.hpp"

namespace KokkosComm::Experimental {
namespace nccl {

namespace KC = KokkosComm;

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
auto broadcast(const ExecSpace &space, const SendView &sv, const RecvView &rv, int root, ncclComm_t comm) -> Req<Nccl> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>,
                "KokkosComm::Experimental::nccl::broadcast: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::broadcast");

  Req<Nccl> req{space.cuda_stream()};
  if (KC::is_contiguous(sv) and KC::is_contiguous(rv)) {
    ncclBroadcast(KC::data_handle(sv), KC::data_handle(rv), KC::span(sv), Impl::datatype_v<ST>, root, comm,
                  space.cuda_stream());
  } else {
    Kokkos::abort("KokkosComm::Experimental::nccl::broadcast: unimplemented for non-contiguous views");
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace nccl
namespace Impl {

template <KokkosView SendView, KokkosView RecvView>
struct Broadcast<SendView, RecvView, Kokkos::Cuda, Nccl> {
  static auto execute(Handle<Kokkos::Cuda, Nccl> &h, const SendView sv, RecvView rv, int root) -> Req<Nccl> {
    return nccl::broadcast(h.space(), sv, rv, root, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
