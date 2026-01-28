// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "mpi_space.hpp"
#include "req.hpp"

#include <KokkosComm/impl/host_staging.hpp>
#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace mpi {

template <KokkosExecutionSpace ExecSpace, KokkosView View>
auto ibroadcast(const ExecSpace& space, View& v, int root, MPI_Comm comm) -> Req<MpiSpace> {
  using T = typename View::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::mpi::ibroadcast");
  fail_if(!is_contiguous(v), "KokkosComm::mpi::ibroadcast: unimplemented for non-contiguous views");

  Req<MpiSpace> req;
#if defined(KOKKOSCOMM_ENABLE_GPU_AWARE_MPI)
  // Sync: Work in space may have been used to produce view data.
  space.fence("fence before GPU-aware `MPI_Ibcast`");
  MPI_Ibcast(data_handle(v), span(v), datatype<MpiSpace, T>(), root, comm, &req.mpi_request());
  req.extend_view_lifetime(v);
#else
  auto host_v = KokkosComm::Impl::stage_for(v);
  // Sync: Ensure that `host_v` is done being copied on the host
  space.fence("fence host staging before `MPI_Ibcast`");
  MPI_Ibcast(data_handle(host_v), span(host_v), datatype<MpiSpace, T>(), root, comm, &req.mpi_request());
  // Implicitly extends lifetimes of `host_v` and `v` due to lambda capture
  req.call_after_mpi_wait([=]() {
    KokkosComm::Impl::copy_back(space, v, host_v);
    space.fence("fence copy back after `MPI_Ibcast`");
  });
#endif

  Kokkos::Tools::popRegion();
  return req;
}

template <KokkosExecutionSpace ExecSpace, KokkosView View>
void broadcast(ExecSpace const& space, View const& v, int root, MPI_Comm comm) {
  using T = typename View::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::mpi::broadcast");
  fail_if(!is_contiguous(v), "KokkosComm::mpi::broadcast: unimplemented for non-contiguous views");

#if defined(KOKKOSCOMM_ENABLE_GPU_AWARE_MPI)
  // Sync: Work in space may have been used to produce view data
  space.fence("fence before GPU-aware `MPI_Bcast`");
  MPI_Bcast(data_handle(v), span(v), datatype<MpiSpace, T>(), root, comm);
#else
  auto host_v = KokkosComm::Impl::stage_for(v);
  space.fence("fence host staging before `MPI_Bcast`");
  MPI_Bcast(data_handle(host_v), span(host_v), datatype<MpiSpace, T>(), root, comm);
  KokkosComm::Impl::copy_back(space, v, host_v);
  space.fence("fence copy back after `MPI_Bcast`");
#endif

  Kokkos::Tools::popRegion();
}

template <KokkosView View>
void broadcast(View const& v, int root, MPI_Comm comm) {
  broadcast(Kokkos::DefaultExecutionSpace{}, v, root, comm);
}

}  // namespace mpi
namespace Experimental::Impl {

template <KokkosView View, KokkosExecutionSpace ExecSpace>
struct Broadcast<View, ExecSpace, MpiSpace> {
  static auto execute(Handle<ExecSpace, MpiSpace>& h, View& v, int root) -> Req<MpiSpace> {
    return KokkosComm::mpi::ibroadcast(h.space(), v, root, h.comm());
  }
};

}  // namespace Experimental::Impl
}  // namespace KokkosComm
