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
auto alltoall(const ExecSpace& space, const SendView& sv, const RecvView& rv, int count, ncclComm_t comm)
    -> Request<NcclSpace> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::Experimental::nccl::alltoall: View value types must be identical");
  Kokkos::Profiling::ScopedRegion region("KokkosComm::Experimental::nccl::alltoall");

  KC_NCCL_FAIL_IF_REQ(!KC::is_contiguous(sv) or !KC::is_contiguous(rv), ErrorCode::NotSupported);

  Request<NcclSpace> req(comm);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
  KC_NCCL_CHECK_REQ(ncclAlltoAll(
      KC::data_handle(sv), KC::data_handle(rv), count, datatype<NcclSpace, ST>(), comm, space.cuda_stream()
  ));
#else
  int n_pes;
  KC_NCCL_CHECK_REQ(ncclCommCount(comm, &n_pes));
  KC_NCCL_CHECK_REQ(ncclGroupStart());
  // Always close the group, even if posting a send/recv failed, before reporting the error
  auto post_all = [&]() -> ncclResult_t {
    for (int r = 0; r < n_pes; ++r) {
      ncclResult_t err =
          ncclSend(KC::data_handle(sv) + r * count, count, datatype<NcclSpace, ST>(), r, comm, space.cuda_stream());
      if (err != ncclSuccess) return err;
      err = ncclRecv(KC::data_handle(rv) + r * count, count, datatype<NcclSpace, ST>(), r, comm, space.cuda_stream());
      if (err != ncclSuccess) return err;
    }
    return ncclSuccess;
  };
  const ncclResult_t post_err = post_all();
  const ncclResult_t end_err  = ncclGroupEnd();
  KC_NCCL_CHECK_REQ(post_err);
  KC_NCCL_CHECK_REQ(end_err);
#endif
  KC_CUDA_CHECK_REQ(req.capture_stream_state(space.cuda_stream()));
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  return req;
}

}  // namespace nccl
namespace Impl {

template <KokkosView SendView, MutKokkosView RecvView>
struct AllToAll<SendView, RecvView, Kokkos::Cuda, NcclSpace> {
  static auto execute(Communicator<NcclSpace, Kokkos::Cuda>& h, const SendView sv, RecvView rv, int count)
      -> Request<NcclSpace> {
    return nccl::alltoall(h.exec(), sv, rv, count, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
