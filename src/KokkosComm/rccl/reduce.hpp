// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>
#include <rccl/rccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/reduction_op.hpp>
#include "rccl_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

#include "impl/pack_traits.hpp"

namespace KokkosComm::Experimental {
namespace rccl {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, MutKokkosView RecvView>
auto reduce(
    const ExecSpace& space, const SendView& sv, RecvView& rv, ncclRedOp_t op, int root, int rank, ncclComm_t comm
) -> Request<RcclSpace> {
  using ST         = typename SendView::non_const_value_type;
  using RT         = typename RecvView::non_const_value_type;
  using SendPacker = typename Impl::PackTraits<SendView>::packer_type;
  using RecvPacker = typename Impl::PackTraits<RecvView>::packer_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::Experimental::rccl::reduce: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::rccl::reduce");

  Request<RcclSpace> req;
  if (is_contiguous(sv)) {
    if (rank != root and is_contiguous(rv)) {
      ncclReduce(
          data_handle(sv), data_handle(rv), span(sv), datatype_for<RcclSpace>(sv), op, root, comm, space.hip_stream()
      );
      req.capture_stream_state(space.hip_stream());
    } else {
      auto pckd_rv = RecvPacker::allocate_packed_for(space, "pckd_rv", rv);
      ncclReduce(
          data_handle(sv), data_handle(pckd_rv.view_), span(sv), datatype_for<RcclSpace>(sv), op, root, comm,
          space.hip_stream()
      );
      req.capture_stream_state(space.hip_stream());
      req.add_callback([space, rv, pckd_rv]() {
        RecvPacker::unpack_into(space, rv, pckd_rv.view_);
        space.fence("fence `pckd_rv` unpacking after RCCL call");
      });
    }
  } else {
    auto pckd_sv = SendPacker::pack(space, "pckd_sv", sv);
    if (rank != root and is_contiguous(rv)) {
      ncclReduce(
          data_handle(pckd_sv.view_), data_handle(rv), pckd_sv.count_, pckd_sv.datatype_, op, root, comm,
          space.hip_stream()
      );
      req.capture_stream_state(space.hip_stream());
    } else {
      auto pckd_rv = RecvPacker::allocate_packed_for(space, "pckd_rv", rv);
      ncclReduce(
          data_handle(pckd_sv.view_), data_handle(pckd_rv.view_), pckd_sv.count_, pckd_sv.datatype_, op, root, comm,
          space.hip_stream()
      );
      req.capture_stream_state(space.hip_stream());
      req.add_callback([space, rv, pckd_rv]() {
        RecvPacker::unpack_into(space, rv, pckd_rv.view_);
        space.fence("fence `pckd_rv` unpacking after RCCL call");
      });
    }
    req.extend_view_lifetime(pckd_sv.view_);
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace rccl
namespace Impl {

template <KokkosView SendView, MutKokkosView RecvView, ReductionOperator RedOp>
struct Reduce<SendView, RecvView, RedOp, Kokkos::HIP, RcclSpace> {
  static auto execute(Communicator<RcclSpace, Kokkos::HIP>& h, const SendView sv, RecvView rv, int root)
      -> Request<RcclSpace> {
    return rccl::reduce(h.exec(), sv, rv, reduction_op<RcclSpace, RedOp>(), root, h.rank(), h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
