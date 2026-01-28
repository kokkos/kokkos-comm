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
#include "impl/pack_traits.hpp"
#include "impl/error_handling.hpp"

namespace KokkosComm {
namespace mpi {

template <KokkosExecutionSpace ExecSpace, KokkosView SView, KokkosView RView>
auto ialltoall(const ExecSpace& space, const SView sv, RView rv, int count, MPI_Comm comm) -> Req<MpiSpace> {
  using ST = typename SView::non_const_value_type;
  using RT = typename RView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::mpi::ialltoall: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::mpi::ialltoall");
  fail_if(!is_contiguous(sv) || !is_contiguous(rv),
          "KokkosComm::mpi::ialltoall: unimplemented for non-contiguous views");

  Req<MpiSpace> req;
#if defined(KOKKOSCOMM_ENABLE_GPU_AWARE_MPI)
  // Sync: Work in space may have been used to produce view data.
  space.fence("fence before non-blocking all-gather");
  // All ranks send/recv same count
  MPI_Ialltoall(data_handle(sv), count, datatype<MpiSpace, ST>(), data_handle(rv), count, datatype<MpiSpace, RT>(),
                comm, &req.mpi_request());
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);
#else
  auto host_sv = KokkosComm::Impl::stage_for(sv);
  auto host_rv = KokkosComm::Impl::stage_for(rv);
  space.fence("fence host staging before `MPI_Ialltoall`");
  MPI_Ialltoall(data_handle(host_sv), count, datatype<MpiSpace, ST>(), data_handle(host_rv), count,
                datatype<MpiSpace, RT>(), comm, &req.mpi_request());
  // Implicitly extends lifetimes of `host_rv` and `rv` due to lambda capture
  req.call_after_mpi_wait([=]() {
    KokkosComm::Impl::copy_back(space, rv, host_rv);
    space.fence("fence copy back after `MPI_Ialltoall`");
  });
  req.extend_view_lifetime(host_sv);
  req.extend_view_lifetime(sv);
#endif

  Kokkos::Tools::popRegion();
  return req;
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void alltoall(const ExecSpace& space, const SendView& sv, size_t s_cnt, RecvView& rv, size_t r_cnt, MPI_Comm comm) {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::mpi::alltoall: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::mpi::alltoall");
  fail_if(!is_contiguous(sv) || !is_contiguous(rv),
          "KokkosComm::mpi::alltoall: unimplemented for non-contiguous views");

  const int size = [&]() {
    int tmp;
    MPI_Comm_size(comm, &tmp);
    return tmp;
  }();
  fail_if(s_cnt * size > extent(sv, 0),
          "KokkosComm::mpi::alltoall: send count * comm size is greater than send view size");
  fail_if(r_cnt * size > extent(rv, 0),
          "KokkosComm::mpi::alltoall: receive count * comm size is greater than receive view size");

#if defined(KOKKOSCOMM_ENABLE_GPU_AWARE_MPI)
  // Sync: Work in space may have been used to produce view data.
  space.fence("fence before GPU-aware `MPI_Alltoall`");
  MPI_Alltoall(data_handle(sv), s_cnt, datatype<MpiSpace, ST>(), data_handle(rv), r_cnt, datatype<MpiSpace, RT>(),
               comm);
#else
  auto host_sv = KokkosComm::Impl::stage_for(sv);
  auto host_rv = KokkosComm::Impl::stage_for(rv);
  space.fence("fence host staging before `MPI_Alltoall`");
  MPI_Alltoall(data_handle(host_sv), s_cnt, datatype<MpiSpace, ST>(), data_handle(host_rv), r_cnt,
               datatype<MpiSpace, RT>(), comm);
  KokkosComm::Impl::copy_back(space, rv, host_rv);
  space.fence("fence copy back after `MPI_Alltoall`");
#endif

  Kokkos::Tools::popRegion();
}

// In-place alltoall
template <KokkosExecutionSpace ExecSpace, KokkosView View>
void alltoall(const ExecSpace& space, View& v, size_t cnt, MPI_Comm comm) {
  using T = typename View::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::mpi::alltoall");
  fail_if(!is_contiguous(v), "KokkosComm::mpi::alltoall: unimplemented for non-contiguous views");

  const int size = [&]() {
    int tmp;
    MPI_Comm_size(comm, &tmp);
    return tmp;
  }();
  fail_if(cnt * size > extent(v, 0), "KokkosComm::mpi::alltoall: count * comm size is greater than view size");

#if defined(KOKKOSCOMM_ENABLE_GPU_AWARE_MPI)
  // Sync: Work in space may have been used to produce view data.
  space.fence("fence before GPU-aware in-place `MPI_Alltoall`");
  MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, data_handle(v), cnt, datatype<MpiSpace, T>(), comm);
#else
  auto host_v = KokkosComm::Impl::stage_for(v);
  space.fence("fence host staging before in-place `MPI_Alltoall`");
  MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, data_handle(host_v), cnt, datatype<MpiSpace, T>(), comm);
  KokkosComm::Impl::copy_back(space, v, host_v);
  space.fence("fence copy back after in-place `MPI_Alltoall`");
#endif

  Kokkos::Tools::popRegion();
}

}  // namespace mpi
namespace Experimental::Impl {

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace>
struct AllToAll<SendView, RecvView, ExecSpace, MpiSpace> {
  static auto execute(Handle<ExecSpace, MpiSpace>& h, const SendView sv, RecvView rv, int count) -> Req<MpiSpace> {
    return mpi::ialltoall(h.space(), sv, rv, count, h.mpi_comm());
  }
};

}  // namespace Experimental::Impl
}  // namespace KokkosComm
