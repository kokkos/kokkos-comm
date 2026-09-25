// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

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
auto allreduce(const ExecSpace& space, const SendView& sv, const RecvView& rv, ncclRedOp_t op, ncclComm_t comm)
    -> Request<RcclSpace> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(
      std::is_same_v<ST, RT>, "KokkosComm::Experimental::rccl::allreduce: View value types must be identical"
  );
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::rccl::allreduce");

  Request<RcclSpace> req;
  if (is_contiguous(sv) and is_contiguous(rv)) {
    ncclAllReduce(
        data_handle(sv), data_handle(rv), span(sv), datatype_for<RcclSpace>(sv), op, comm, space.hip_stream()
    );
    req.capture_stream_state(space.hip_stream());
  } else {
    Kokkos::abort("KokkosComm::Experimental::rccl::allreduce: unimplemented for non-contiguous Views");
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace rccl
namespace Impl {

template <KokkosView SendView, MutKokkosView RecvView, ReductionOperator RedOp>
struct AllReduce<SendView, RecvView, RedOp, Kokkos::HIP, RcclSpace> {
  static auto execute(Communicator<RcclSpace, Kokkos::HIP>& h, const SendView sv, RecvView rv) -> Request<RcclSpace> {
    return rccl::allreduce(h.exec(), sv, rv, reduction_op<RcclSpace, RedOp>(), h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
