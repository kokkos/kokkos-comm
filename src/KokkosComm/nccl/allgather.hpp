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
auto allgather(const ExecSpace& space, const SendView& sv, const RecvView& rv, ncclComm_t comm) -> Request<NcclSpace> {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;

  constexpr const char* fn = "KokkosComm::Experimental::nccl::allgather";
  Kokkos::Tools::pushRegion(fn);

  KokkosComm::Impl::checks::static_assert_dtype_match(sv, rv);
  KokkosComm::Impl::checks::static_assert_rank_match_allgather(sv, rv);
  int comm_size;
  if constexpr (metadata_checks) KC_NCCL_CHECK(ncclCommCount(comm, &comm_size));
  KokkosComm::Impl::checks::fail_if_size_mismatch_allgather(sv, rv, fn, comm_size);
  KokkosComm::Impl::checks::fail_if_noncontiguous(sv, fn);
  KokkosComm::Impl::checks::fail_if_noncontiguous(rv, fn);

  Request<NcclSpace> req;
  KC_NCCL_CHECK(ncclAllGather(
      KC::data_handle(sv), KC::data_handle(rv), KC::span(sv), datatype<NcclSpace, ST>(), comm, space.cuda_stream()
  ));
  req.capture_stream_state(space.cuda_stream());

  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
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
