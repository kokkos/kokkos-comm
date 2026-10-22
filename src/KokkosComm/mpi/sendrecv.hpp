// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/fwd.hpp>
#include "mpi_space.hpp"
#include "communicator.hpp"
#include "request.hpp"

#include "impl/pack_traits.hpp"
#include "impl/tags.hpp"

namespace KokkosComm {
namespace mpi {

template <KokkosExecutionSpace Exec, KokkosView SendV, MutKokkosView RecvV>
auto isendrecv(Exec const& exec, SendV const& sv, int dst, int stag, RecvV const& rv, int src, int rtag, MPI_Comm comm)
    -> Request<MpiSpace> {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::isendrecv");

  Request<MpiSpace> req;

  const void* sbuf = data_handle(sv);
  auto scnt        = span(sv);
  auto sdtype      = datatype_for<MpiSpace>(sv);
  if (!is_contiguous(sv)) {
    using SendPckr = typename Impl::PackTraits<SendV>::packer_type;
    auto pckd_sv   = SendPckr::pack(exec, "pckd_sv", sv);
    sbuf           = data_handle(pckd_sv.view);
    scnt           = pckd_sv.count;
    sdtype         = pckd_sv.datatype;
    req.extend_view_lifetime(pckd_sv.view);
  }

  void* rbuf  = data_handle(rv);
  auto rcnt   = span(rv);
  auto rdtype = datatype_for<MpiSpace>(rv);
  if (!is_contiguous(rv)) {
    using RecvPckr = typename Impl::PackTraits<RecvV>::packer_type;
    auto pckd_rv   = RecvPckr::allocate_packed_for(exec, "pckd_rv", rv);
    rbuf           = data_handle(pckd_rv.view);
    rcnt           = pckd_rv.count;
    rdtype         = pckd_rv.datatype;
    // Implicitly extends `pckd_rv.view` and `rv` lifetime due to lambda capture
    req.add_callback([exec, rv, pckd_rv]() {
      RecvPckr::unpack_into(exec, rv, pckd_rv.view);
      exec.fence("fence unpacking after MPI_Isendrecv");
    });
  }

  exec.fence("fence before MPI_Isendrecv");
  MPI_Isendrecv(sbuf, scnt, sdtype, dst, stag, rbuf, rcnt, rdtype, src, rtag, comm, req.request_ptr());
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}
}  // namespace mpi
namespace Impl {

template <KokkosExecutionSpace Exec, KokkosView SendV, MutKokkosView RecvV>
struct Exchange<MpiSpace, Exec, SendV, RecvV> {
  static auto execute(Communicator<MpiSpace, Exec>& comm, SendV const& sv, int dst, const RecvV& rv, int src)
      -> Request<MpiSpace> {
    return mpi::isendrecv(comm.exec(), sv, dst, Impl::POINTTOPOINT_TAG, rv, src, Impl::POINTTOPOINT_TAG, comm.comm());
  }
};

}  // namespace Impl
}  // namespace KokkosComm
