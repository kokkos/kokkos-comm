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
#include "KokkosComm/impl/metadata_checks.hpp"
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

  constexpr const char* fn = "KokkosComm::Experimental::nccl::alltoall";
  Kokkos::Tools::pushRegion(fn);

  KokkosComm::Impl::checks::static_assert_dtype_match(sv, rv);
  KokkosComm::Impl::checks::static_assert_rank_match(sv, rv);
  KokkosComm::Impl::checks::fail_if_extents_mismatch(sv, rv, fn);
  int comm_size;
  if constexpr (metadata_checks) KC_NCCL_CHECK(ncclCommCount(comm, &comm_size));
  KokkosComm::Impl::checks::fail_if_count_mismatch_alltoall(sv, count, comm_size, fn);
  KokkosComm::Impl::checks::fail_if_count_mismatch_alltoall(rv, count, comm_size, fn);

  KokkosComm::Impl::checks::fail_if_noncontiguous(sv, fn);
  KokkosComm::Impl::checks::fail_if_noncontiguous(rv, fn);

  Request<NcclSpace> req;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
  KC_NCCL_CHECK(ncclAlltoAll(
      KC::data_handle(sv), KC::data_handle(rv), count, datatype<NcclSpace, ST>(), comm, space.cuda_stream()
  ));
#else
  int n_pes;
  KC_NCCL_CHECK(ncclCommCount(comm, &n_pes));
  KC_NCCL_CHECK(ncclGroupStart());
  for (int r = 0; r < n_pes; ++r) {
    KC_NCCL_CHECK(
        ncclSend(KC::data_handle(sv) + r * count, count, datatype<NcclSpace, ST>(), r, comm, space.cuda_stream())
    );
    KC_NCCL_CHECK(
        ncclRecv(KC::data_handle(rv) + r * count, count, datatype<NcclSpace, ST>(), r, comm, space.cuda_stream())
    );
  }
  KC_NCCL_CHECK(ncclGroupEnd());
#endif
  req.capture_stream_state(space.cuda_stream());

  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
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
