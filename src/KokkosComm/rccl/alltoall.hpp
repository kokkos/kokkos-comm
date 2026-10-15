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

namespace KokkosComm::Experimental {
namespace rccl {

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, MutKokkosView RecvView>
auto alltoall(const ExecSpace& space, const SendView& sv, const RecvView& rv, int count, ncclComm_t comm)
    -> Request<RcclSpace> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::Experimental::rccl::alltoall: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::rccl::alltoall");

  Request<RcclSpace> req;
  if (is_contiguous(sv) and is_contiguous(rv)) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 3)
    // NCCL-compatible call
    ncclAlltoAll(data_handle(sv), data_handle(rv), count, datatype_for<RcclSpace>(sv), comm, space.hip_stream());
#else
    ncclAllToAll(data_handle(sv), data_handle(rv), count, datatype_for<RcclSpace>(sv), comm, space.hip_stream());
#endif
    req.capture_stream_state(space.hip_stream());
  } else {
    Kokkos::abort("KokkosComm::Experimental::rccl::alltoall: unimplemented for non-contiguous views");
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace rccl
namespace Impl {

template <KokkosView SendView, MutKokkosView RecvView>
struct AllToAll<SendView, RecvView, Kokkos::HIP, RcclSpace> {
  static auto execute(Communicator<RcclSpace, Kokkos::HIP>& h, const SendView sv, RecvView rv, int count)
      -> Request<RcclSpace> {
    return rccl::alltoall(h.exec(), sv, rv, count, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
