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

template <MutKokkosView View>
auto broadcast(const Kokkos::HIP& space, View& v, int root, ncclComm_t comm) -> Request<RcclSpace> {
  using T = typename View::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::rccl::broadcast");

  Request<RcclSpace> req;
  if (is_contiguous(v)) {
    int rank;
    ncclCommUserRank(comm, &rank);
    if (rank == root) {
      ncclBroadcast(
          data_handle(v), data_handle(v), span(v), datatype_for<RcclSpace>(v), root, comm, space.hip_stream()
      );
    } else {
      ncclBroadcast(nullptr, data_handle(v), span(v), datatype_for<RcclSpace>(v), root, comm, space.hip_stream());
    }
    req.capture_stream_state(space.hip_stream());
  } else {
    Kokkos::abort("KokkosComm::Experimental::rccl::broadcast: unimplemented for non-contiguous views");
  }
  req.extend_view_lifetime(v);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace rccl
namespace Impl {

template <MutKokkosView View>
struct Broadcast<View, Kokkos::HIP, RcclSpace> {
  static auto execute(Communicator<RcclSpace, Kokkos::HIP>& h, View v, int root) -> Request<RcclSpace> {
    return rccl::broadcast(h.exec(), v, root, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
