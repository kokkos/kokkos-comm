// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/reduction_op.hpp>
#include "nccl_space.hpp"
#include "handle.hpp"
#include "req.hpp"

#include <KokkosComm/impl/contiguous.hpp>
#include "impl/pack_traits.hpp"

namespace KokkosComm::Experimental {
namespace nccl {

namespace KC = KokkosComm;

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
auto reduce(const ExecSpace &space, const SendView sv, RecvView rv, ncclRedOp_t op, int root, int rank, ncclComm_t comm)
    -> Req<NcclSpace> {
  using SendPacker = typename Impl::PackTraits<SendView>::packer_type;
  using RecvPacker = typename Impl::PackTraits<RecvView>::packer_type;
  using ST         = typename SendView::non_const_value_type;
  using RT         = typename RecvView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::Experimental::nccl::reduce: View value types must be identical");
  static_assert(KC::rank<SendView>() <= 1 and KC::rank<RecvView>() <= 1,
                "KokkosComm::Experimental::nccl::reduce: Views with rank higher than 1 are not supported");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::reduce");

  Req<NcclSpace> req{space.cuda_stream()};
  if (KC::is_contiguous(sv)) {
    if (rank != root and KC::is_contiguous(rv)) {
      ncclReduce(KC::data_handle(sv), KC::data_handle(rv), KC::span(sv), datatype<NcclSpace, ST>(), op, root, comm,
                 space.cuda_stream());
    } else {
      auto pckd_rv = KC::Impl::allocate_contiguous_for(space, "KC::nccl::reduce pckd_rv", rv);
      ncclReduce(KC::data_handle(sv), KC::data_handle(pckd_rv), KC::span(sv), datatype<NcclSpace, ST>(), op, root, comm,
                 space.cuda_stream());
      RecvPacker::unpack_into(space, rv, pckd_rv);
      req.extend_view_lifetime(pckd_rv);
    }
  } else {
    auto send_args = SendPacker::pack(space, sv);
    if (rank != root and KC::is_contiguous(rv)) {
      ncclReduce(KC::data_handle(send_args.view_), KC::data_handle(rv), KC::span(send_args.view_),
                 datatype<NcclSpace, ST>(), op, root, comm, space.cuda_stream());
    } else {
      auto pckd_rv = KC::Impl::allocate_contiguous_for(space, "KC::nccl::reduce pckd_rv", rv);
      ncclReduce(KC::data_handle(send_args.view_), KC::data_handle(pckd_rv), KC::span(send_args.view_),
                 datatype<NcclSpace, ST>(), op, root, comm, space.cuda_stream());
      RecvPacker::unpack_into(space, rv, pckd_rv);
      req.extend_view_lifetime(pckd_rv);
    }
    req.extend_view_lifetime(send_args.view_);
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace nccl
namespace Impl {

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp>
struct Reduce<SendView, RecvView, RedOp, Kokkos::Cuda, NcclSpace> {
  static auto execute(Handle<Kokkos::Cuda, NcclSpace> &h, const SendView sv, RecvView rv, int root) -> Req<NcclSpace> {
    return nccl::reduce(h.space(), sv, rv, reduction_op<NcclSpace, RedOp>(), root, h.rank(), h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
