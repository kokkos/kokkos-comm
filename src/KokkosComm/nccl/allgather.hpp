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

namespace KokkosComm::Experimental {
namespace nccl {

namespace KC = KokkosComm;

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
auto allgather(const ExecSpace& space, const SendView& sv, const RecvView& rv, ncclComm_t comm) -> Request<NcclSpace> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(
      std::is_same_v<ST, RT>, "KokkosComm::Experimental::nccl::allgather: View value types must be identical"
  );
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::allgather");

  Request<NcclSpace> req;
  if (KC::is_contiguous(sv) and KC::is_contiguous(rv)) {
    ncclAllGather(
        KC::data_handle(sv), KC::data_handle(rv), KC::span(sv), datatype<NcclSpace, ST>(), comm, space.cuda_stream()
    );
    req.capture_stream_state(space.cuda_stream());
  } else {
    Kokkos::abort("KokkosComm::Experimental::nccl::allgather: unimplemented for non-contiguous views");
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace nccl
namespace Impl {

template <KokkosView SendView, KokkosView RecvView>
struct AllGather<SendView, RecvView, Kokkos::Cuda, NcclSpace> {
  static auto execute(Communicator<NcclSpace, Kokkos::Cuda>& h, const SendView sv, RecvView rv) -> Request<NcclSpace> {
    return nccl::allgather(h.exec(), sv, rv, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
