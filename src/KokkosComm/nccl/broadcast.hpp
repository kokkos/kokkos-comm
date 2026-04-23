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

template <KokkosView View>
auto broadcast(const Kokkos::Cuda& space, View& v, int root, ncclComm_t comm) -> Request<NcclSpace> {
  using T = typename View::non_const_value_type;
  static_assert(
      KC::rank<View>() <= 1,
      "KokkosComm::Experimental::nccl::broadcast: Views with rank higher than 1 are not supported"
  );
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::broadcast");

  Request<NcclSpace> req;
  if (KC::is_contiguous(v)) {
    ncclBcast(KC::data_handle(v), KC::span(v), datatype<NcclSpace, T>(), root, comm, space.cuda_stream());
    req.capture_stream_state(space.cuda_stream());
  } else {
    Kokkos::abort("KokkosComm::Experimental::nccl::broadcast: unimplemented for non-contiguous views");
  }
  req.extend_view_lifetime(v);

  Kokkos::Tools::popRegion();
  return req;
}

}  // namespace nccl
namespace Impl {

template <KokkosView View>
struct Broadcast<View, Kokkos::Cuda, NcclSpace> {
  static auto execute(Communicator<NcclSpace, Kokkos::Cuda>& h, View v, int root) -> Request<NcclSpace> {
    return nccl::broadcast(h.exec(), v, root, h.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm::Experimental
