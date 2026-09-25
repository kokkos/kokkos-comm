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

template <MutKokkosView View>
auto broadcast(const Kokkos::Cuda& space, View& v, int root, ncclComm_t comm) -> Request<NcclSpace> {
  using T = typename View::non_const_value_type;

  constexpr const char* fn = "KokkosComm::Experimental::nccl::broadcast";
  Kokkos::Tools::pushRegion(fn);

  KokkosComm::Impl::checks::static_assert_rank_leq_1(v);
  KokkosComm::Impl::checks::fail_if_noncontiguous(v, fn);

  Request<NcclSpace> req;
  KC_NCCL_CHECK(ncclBcast(KC::data_handle(v), KC::span(v), datatype<NcclSpace, T>(), root, comm, space.cuda_stream()));
  req.capture_stream_state(space.cuda_stream());
  req.extend_view_lifetime(v);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace nccl
namespace Impl {

template <MutKokkosView View>
struct Broadcast<View, Kokkos::Cuda, NcclSpace> {
  static auto execute(Communicator<NcclSpace, Kokkos::Cuda>& h, View v, int root) -> Request<NcclSpace> {
    return nccl::broadcast(h.exec(), v, root, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
