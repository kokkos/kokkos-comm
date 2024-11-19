//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#pragma once

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/reduction_op.hpp>
#include <KokkosComm/nccl/impl/pack_traits.hpp>
#include <KokkosComm/nccl/impl/types.hpp>

#include <Kokkos_Core.hpp>
#include <nccl.h>

namespace KokkosComm::Experimental::nccl::Impl {

template <ReductionOperator RedOp>
constexpr auto reduction_op() -> ncclRedOp_t {
  if constexpr (std::is_same_v<RedOp, ReduceMaximum>) {
    return ncclMax;
  } else if constexpr (std::is_same_v<RedOp, ReduceMinimum>) {
    return ncclMin;
  } else if constexpr (std::is_same_v<RedOp, ReduceSum>) {
    return ncclSum;
  } else if constexpr (std::is_same_v<RedOp, ReduceProduct>) {
    return ncclProd;
  } else if constexpr (std::is_same_v<RedOp, ReduceAverage>) {
    return ncclAvg;
  } else {
    {
      static_assert(std::is_void_v<RedOp>, "NCCL reduction operator not implemented");
      return ncclMax;  // unreachable
    }
  }
}

template <ReductionOperator RedOp>
inline constexpr ncclRedOp_t reduction_op_v = reduction_op<RedOp>();

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void reduce(const ExecSpace &space, const SendView sv, RecvView rv, ncclRedOp_t op, int root, int rank,
            ncclComm_t comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Experimental::nccl::Impl::reduce");

  using SendPacker = typename PackTraits<SendView>::packer_type;
  using RecvPacker = typename PackTraits<RecvView>::packer_type;

  if (!KokkosComm::is_contiguous(sv)) {
    auto send_args = SendPacker::pack(space, sv);
    space.fence();
    if ((root == rank) && !KokkosComm::is_contiguous(rv)) {
      auto recv_args = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence();
      using SendScalar = typename SendView::non_const_value_type;
      ncclReduce(send_args.view.data(), recv_args.view.data(), send_args.count, send_args.datatype, op, root, comm,
                 space.cuda_stream());
      RecvPacker::unpack_into(space, rv, recv_args.view);
    } else {
      space.fence();  // is this fence necessary?
      ncclReduce(send_args.view.data(), rv.data(), send_args.count, send_args.datatype, op, root, comm,
                 space.cuda_stream());
    }
  } else {
    using SendScalar = typename SendView::value_type;
    if ((root == rank) && !KokkosComm::is_contiguous(rv)) {
      auto recv_args = RecvPacker::allocate_packed_for(space, "reduce recv", rv);
      space.fence();
      ncclReduce(sv.data(), recv_args.view.data(), sv.span(), datatype_v<SendScalar>, op, root, comm,
                 space.cuda_stream());
      RecvPacker::unpack_into(space, rv, recv_args.view);
    } else {
      space.fence();  // is this fence necessary?
      ncclReduce(sv.data(), rv.data(), sv.span(), datatype_v<SendScalar>, op, root, comm, space.cuda_stream());
    }
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::Experimental::nccl::Impl
