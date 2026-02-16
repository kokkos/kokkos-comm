// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>
#include <nccl.h>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "nccl_space.hpp"
#include "handle.hpp"
#include "request.hpp"

#include "impl/pack_traits.hpp"

namespace KokkosComm::Experimental {
namespace nccl {

namespace KC = KokkosComm;

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
auto alltoall(const ExecSpace& space, const SendView& sv, const RecvView& rv, int count, ncclComm_t comm)
    -> Request<NcclSpace> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::Experimental::nccl::alltoall: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::alltoall");

  Request<NcclSpace> req;
  if (KC::is_contiguous(sv) and KC::is_contiguous(rv)) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    ncclAlltoAll(KC::data_handle(sv), KC::data_handle(rv), count, datatype<NcclSpace, ST>(), comm, space.cuda_stream());
#else
    int n_pes;
    ncclCommCount(comm, &n_pes);
    ncclGroupStart();
    for (int r = 0; r < n_pes; ++r) {
      ncclSend(KC::data_handle(sv) + r * count, count, datatype<NcclSpace, ST>(), r, comm, space.cuda_stream());
      ncclRecv(KC::data_handle(rv) + r * count, count, datatype<NcclSpace, ST>(), r, comm, space.cuda_stream());
    }
    ncclGroupEnd();
#endif
    req.capture_stream_state(space.cuda_stream());
  } else {
    Kokkos::abort("KokkosComm::Experimental::nccl::alltoall: unimplemented for non-contiguous views");
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace nccl
namespace Impl {

template <KokkosView SendView, KokkosView RecvView>
struct AllToAll<SendView, RecvView, Kokkos::Cuda, NcclSpace> {
  static auto execute(Handle<Kokkos::Cuda, NcclSpace>& h, const SendView sv, RecvView rv, int count)
      -> Request<NcclSpace> {
    return nccl::alltoall(h.space(), sv, rv, count, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
