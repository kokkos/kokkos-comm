// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/error.hpp>
#include "nccl_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm::Experimental {
namespace nccl {

namespace KC = KokkosComm;

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, MutKokkosView RecvView>
auto allgather(const ExecSpace& space, const SendView& sv, const RecvView& rv, ncclComm_t comm) -> Request<NcclSpace> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(
      std::is_same_v<ST, RT>, "KokkosComm::Experimental::nccl::allgather: View value types must be identical"
  );
  Kokkos::Profiling::ScopedRegion region("KokkosComm::Experimental::nccl::allgather");

  KC_NCCL_FAIL_IF_REQ(!KC::is_contiguous(sv) or !KC::is_contiguous(rv), ErrorCode::NotSupported);

  Request<NcclSpace> req(comm);
  KC_NCCL_CHECK_REQ(ncclAllGather(
      KC::data_handle(sv), KC::data_handle(rv), KC::span(sv), datatype<NcclSpace, ST>(), comm, space.cuda_stream()
  ));
  KC_CUDA_CHECK_REQ(req.capture_stream_state(space.cuda_stream()));
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  return req;
}

}  // namespace nccl
namespace Impl {

template <KokkosView SendView, MutKokkosView RecvView>
struct AllGather<SendView, RecvView, Kokkos::Cuda, NcclSpace> {
  static auto execute(Communicator<NcclSpace, Kokkos::Cuda>& h, const SendView sv, RecvView rv) -> Request<NcclSpace> {
    return nccl::allgather(h.exec(), sv, rv, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
