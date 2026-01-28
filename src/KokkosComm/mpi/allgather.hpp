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

template <KokkosExecutionSpace E, KokkosView SV, KokkosView RV>
auto iallgather(const E& space, const SV& sv, RV& rv, MPI_Comm comm) -> Req<MpiSpace> {
  using ST = typename SV::non_const_value_type;
  using RT = typename RV::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::mpi::iallgather: View value types must be identical");
  Kokkos::Tools::pushRegion("KokkosComm::mpi::iallgather");
  fail_if(!is_contiguous(sv) || !is_contiguous(rv),
          "KokkosComm::mpi::iallgather: unimplemented for non-contiguous views");

  fail_if(span(sv) == span(rv), "KokkosComm::mpi::iallgather: all ranks must send & receive the same count");
  const int cnt = span(sv);

  Req<MpiSpace> req;
#if defined(KOKKOSCOMM_ENABLE_GPU_AWARE_MPI)
  // Sync: Work in space may have been used to produce view data.
  space.fence("fence before GPU-aware `MPI_Iallgather`");
  MPI_Iallgather(data_handle(sv), cnt, datatype<MpiSpace, ST>, data_handle(rv), cnt, datatype<MpiSpace, RT>, comm,
                 &req.mpi_request());
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);
#else
  auto host_sv = KokkosComm::Impl::stage_for(sv);
  auto host_rv = KokkosComm::Impl::stage_for(rv);
  space.fence("fence host staging before `MPI_Iallgather`");
  MPI_Iallgather(data_handle(host_sv), cnt, datatype<MpiSpace, ST>, data_handle(host_rv), cnt, datatype<MpiSpace, RT>,
                 comm, &req.mpi_request());
  // Implicitly extends lifetimes of `host_rv` and `rv` due to lambda capture
  req.call_after_mpi_wait([=]() {
    KokkosComm::Impl::copy_back(space, rv, host_rv);
    space.fence("fence copy back after `MPI_Iallgather`");
  });
  req.extend_view_lifetime(host_sv);
  req.extend_view_lifetime(sv);
#endif

  Kokkos::Tools::popRegion();
  return req;
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void allgather(const ExecSpace& space, const SendView& sv, const RecvView& rv, MPI_Comm comm) {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::mpi::allgather");
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::mpi::allgather: View value types must be identical");
  fail_if(!is_contiguous(sv) || !is_contiguous(rv),
          "KokkosComm::mpi::allgather: unimplemented for non-contiguous Views");

  fail_if(span(sv) == span(rv), "KokkosComm::mpi::allgather: all ranks must send & receive the same count");
  const int cnt = span(sv);

#if defined(KOKKOSCOMM_ENABLE_GPU_AWARE_MPI)
  // Sync: Work in space may have been used to produce send view data
  space.fence("fence before GPU-aware `MPI_Allgather`");
  MPI_Allgather(data_handle(sv), cnt, datatype<MpiSpace, ST>(), data_handle(rv), cnt, datatype<MpiSpace, RT>(), comm);
#else
  auto host_sv = KokkosComm::Impl::stage_for(sv);
  auto host_rv = KokkosComm::Impl::stage_for(rv);
  space.fence("fence host staging before `MPI_Allgather`");
  MPI_Allgather(data_handle(host_sv), cnt, datatype<MpiSpace, ST>(), data_handle(host_rv), cnt,
                datatype<MpiSpace, RT>(), comm);
  KokkosComm::Impl::copy_back(space, rv, host_rv);
  space.fence("fence copy back after `MPI_Allgather`");
#endif

  Kokkos::Tools::popRegion();
}

template <KokkosView SendView, KokkosView RecvView>
void allgather(const SendView& sv, const RecvView& rv, MPI_Comm comm) {
  allgather(Kokkos::DefaultExecutionSpace{}, sv, rv, comm);
}

// In-place allgather
template <KokkosExecutionSpace ExecSpace, KokkosView View>
void allgather(const ExecSpace& space, View& v, size_t cnt, MPI_Comm comm) {
  using T = typename View::non_const_value_type;
  Kokkos::Tools::pushRegion("KokkosComm::mpi::allgather");
  fail_if(!is_contiguous(v), "KokkosComm::mpi::allgather: unimplemented for non-contiguous view");

#if defined(KOKKOSCOMM_ENABLE_GPU_AWARE_MPI)
  // Sync: Work in space may have been used to produce send view data
  space.fence("fence before GPU-aware in-place `MPI_Allgather`");
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, data_handle(v), cnt, datatype<MpiSpace, T>(), comm);
#else
  auto host_v = KokkosComm::Impl::stage_for(v);
  space.fence("fence host staging before in-place `MPI_Allgather`");
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, data_handle(host_v), cnt, datatype<MpiSpace, T>(), comm);
  KokkosComm::Impl::copy_back(space, v, host_v);
  space.fence("fence copy back after in-place `MPI_Allgather`");
#endif

  Kokkos::Tools::popRegion();
}

// In-place allgather
template <KokkosView View>
void allgather(View& v, size_t cnt, MPI_Comm comm) {
  allgather(Kokkos::DefaultExecutionSpace{}, v, cnt, comm);
}

}  // namespace mpi
namespace Experimental::Impl {

template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace>
struct AllGather<SendView, RecvView, ExecSpace, MpiSpace> {
  static auto execute(Handle<ExecSpace, MpiSpace>& h, const SendView sv, RecvView rv) -> Req<MpiSpace> {
    return mpi::iallgather(h.space(), sv, rv, h.comm());
  }
};

}  // namespace Experimental::Impl
}  // namespace KokkosComm
