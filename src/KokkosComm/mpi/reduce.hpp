// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include <KokkosComm/reduction_op.hpp>
#include "mpi_space.hpp"
#include "req.hpp"

#include "impl/error_handling.hpp"
#include "impl/pack_traits.hpp"

namespace KokkosComm {
namespace mpi {

template <KokkosExecutionSpace ExecSpace, KokkosView SView, KokkosView RView>
auto ireduce(const ExecSpace& space, const SView& sv, RView& rv, MPI_Op op, int root, MPI_Comm comm) -> Req<MpiSpace> {
// FIXME_EXTERNAL #215
#if defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
  // Unsupported if running Open MPI and Views are in CUDA or HIP execution spaces
  static_assert(
#if defined(KOKKOS_ENABLE_CUDA)
      not std::is_same_v<typename SView::execution_space, Kokkos::Cuda> and
          not std::is_same_v<typename RView::execution_space, Kokkos::Cuda>,
#elif defined(KOKKOS_ENABLE_HIP)
      not std::is_same_v<typename SView::execution_space, Kokkos::HIP> and
          not std::is_same_v<typename RView::execution_space, Kokkos::HIP>,
#endif
      "KokkosComm::mpi::ireduce: Unsupported with Open MPI + Kokkos CUDA/HIP backend");
#endif
  using ST   = typename SView::non_const_value_type;
  using RT   = typename RView::non_const_value_type;
  using SPkr = typename Impl::PackTraits<SView>::packer_type;
  using RPkr = typename Impl::PackTraits<RView>::packer_type;
  static_assert(std::is_same_v<ST, RT>, "KokkosComm::mpi::ireduce: View value types must be identical");
  static_assert(rank<SView>() <= 1 and rank<RView>() <= 1,
                "KokkosComm::mpi::ireduce: Views with rank higher than 1 are not supported");
  Kokkos::Tools::pushRegion("KokkosComm::mpi::ireduce");

  const int rank = [=]() {
    int _r;
    MPI_Comm_rank(comm, &_r);
    return _r;
  }();

  Req<MpiSpace> req;
  if (is_contiguous(sv)) {
    if (rank == root and not is_contiguous(rv)) {
      auto pkd_rv = RPkr::allocate_packed_for(space, "KC::mpi::ireduce c_sv pkd_rv", rv);
      space.fence("fence `pkd_rv` packing before MPI call");
      MPI_Ireduce(data_handle(sv), data_handle(pkd_rv.view), span(sv), datatype<MpiSpace, ST>(), op, root, comm,
                  &req.mpi_request());
      // Implicitly extend `pkd_rv` lifetime because of lambda capture
      req.call_after_mpi_wait([=]() {
        RPkr::unpack_into(space, rv, pkd_rv.view);
        space.fence("fence `pkd_rv` unpacking after MPI call");
      });
    } else {
      space.fence("fence before MPI call");
      MPI_Ireduce(data_handle(sv), data_handle(rv), span(sv), datatype<MpiSpace, ST>(), op, root, comm,
                  &req.mpi_request());
    }
  } else {
    auto pkd_sv = SPkr::pack(space, "pkd_sv", sv);
    space.fence("fence `pkd_sv` packing before MPI call");
    if (rank == root and not is_contiguous(rv)) {
      auto pkd_rv = RPkr::allocate_packed_for(space, "KC::mpi::ireduce nc_sv pkd_rv", rv);
      space.fence("fence `pkd_rv` packing before MPI call");
      MPI_Ireduce(data_handle(pkd_sv.view), data_handle(pkd_rv.view), pkd_sv.count, pkd_sv.datatype, op, root, comm,
                  &req.mpi_request());
      // Implicitly extend `pkd_rv` lifetime because of lambda capture
      req.call_after_mpi_wait([=]() {
        RPkr::unpack_into(space, rv, pkd_rv.view);
        space.fence("fence `pkd_rv` unpacking after MPI call");
      });
    } else {
      MPI_Ireduce(data_handle(pkd_sv.view), data_handle(rv), pkd_sv.count, pkd_sv.datatype, op, root, comm,
                  &req.mpi_request());
    }
    req.extend_view_lifetime(pkd_sv.view);
  }
  req.extend_view_lifetime(sv);
  req.extend_view_lifetime(rv);

  Kokkos::Tools::popRegion();
  return req;
}

template <KokkosView SendView, KokkosView RecvView>
void reduce(const SendView& sv, RecvView& rv, MPI_Op op, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::reduce");

  KokkosComm::mpi::fail_if(!KokkosComm::is_contiguous(sv) || !KokkosComm::is_contiguous(rv),
                           "only contiguous views supported for low-level reduce");

  using SendScalar = typename SendView::non_const_value_type;
  MPI_Reduce(KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), KokkosComm::span(sv),
             datatype<MpiSpace, SendScalar>(), op, root, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void reduce(const ExecSpace& space, const SendView& sv, RecvView& rv, MPI_Op op, int root, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::reduce");

  const int rank = [=]() -> int {
    int _r;
    MPI_Comm_rank(comm, &_r);
    return _r;
  }();

  using SendPacker = typename Impl::PackTraits<SendView>::packer_type;
  using RecvPacker = typename Impl::PackTraits<RecvView>::packer_type;

  if (!KokkosComm::is_contiguous(sv)) {
    auto sendArgs = SendPacker::pack(space, "pkd_sv", sv);
    if ((root == rank) && !KokkosComm::is_contiguous(rv)) {
      auto recvArgs = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence("fence allocation before MPI call");
      MPI_Reduce(KokkosComm::data_handle(sendArgs.view), KokkosComm::data_handle(recvArgs.view), sendArgs.count,
                 sendArgs.datatype, op, root, comm);
      RecvPacker::unpack_into(space, rv, recvArgs.view);
    } else {
      space.fence("fence packing before MPI call");
      MPI_Reduce(KokkosComm::data_handle(sendArgs.view), KokkosComm::data_handle(rv), sendArgs.count, sendArgs.datatype,
                 op, root, comm);
    }
  } else {
    using SendScalar = typename SendView::value_type;
    if ((root == rank) && !KokkosComm::is_contiguous(rv)) {
      auto recvArgs = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence("fence allocation before MPI call");
      MPI_Reduce(KokkosComm::data_handle(sv), KokkosComm::data_handle(recvArgs.view), KokkosComm::span(sv),
                 datatype<MpiSpace, SendScalar>(), op, root, comm);
      RecvPacker::unpack_into(space, rv, recvArgs.view);
    } else {
      space.fence("fence space before MPI call");
      MPI_Reduce(KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), KokkosComm::span(sv),
                 datatype<MpiSpace, SendScalar>(), op, root, comm);
    }
  }

  Kokkos::Tools::popRegion();
}

}  // namespace mpi
namespace Experimental::Impl {

template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp, KokkosExecutionSpace ExecSpace>
struct Reduce<SendView, RecvView, RedOp, ExecSpace, MpiSpace> {
  static auto execute(Handle<ExecSpace, MpiSpace>& h, const SendView sv, RecvView rv, int root) -> Req<MpiSpace> {
    return mpi::ireduce(h.space(), sv, rv, reduction_op<MpiSpace, RedOp>(), root, h.mpi_comm());
  }
};

}  // namespace Experimental::Impl
}  // namespace KokkosComm
