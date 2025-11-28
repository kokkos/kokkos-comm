// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>
#ifdef KOKKOSCOMM_ENABLE_NCCL
#include <nccl.h>
#endif

#include <KokkosComm/concepts.hpp>
#include "mpi/mpi_space.hpp"
#ifdef KOKKOSCOMM_ENABLE_NCCL
#include "nccl/nccl_space.hpp"
#endif

namespace KokkosComm {
namespace {

// clang-format off
#define DECL_REDUCTION_OP_FOR(operator) \
  struct operator;                      \
  template <> struct Impl::is_reduction_operator<operator> : public std::true_type {}
// clang-format on

}  // namespace

DECL_REDUCTION_OP_FOR(BAnd);
DECL_REDUCTION_OP_FOR(BOr);
DECL_REDUCTION_OP_FOR(BXor);
DECL_REDUCTION_OP_FOR(LAnd);
DECL_REDUCTION_OP_FOR(LOr);
DECL_REDUCTION_OP_FOR(LXor);
DECL_REDUCTION_OP_FOR(Max);
DECL_REDUCTION_OP_FOR(MaxLoc);
DECL_REDUCTION_OP_FOR(Min);
DECL_REDUCTION_OP_FOR(MinLoc);
DECL_REDUCTION_OP_FOR(Sum);
DECL_REDUCTION_OP_FOR(Prod);
DECL_REDUCTION_OP_FOR(Average);

namespace Impl {

template <ReductionOperator RO>
constexpr auto mpi_reduction_op() -> MPI_Op {
  if constexpr (std::is_same_v<RO, BAnd>) {
    return MPI_BAND;
  } else if constexpr (std::is_same_v<RO, BOr>) {
    return MPI_BOR;
  } else if constexpr (std::is_same_v<RO, BXor>) {
    return MPI_BXOR;
  } else if constexpr (std::is_same_v<RO, LAnd>) {
    return MPI_LAND;
  } else if constexpr (std::is_same_v<RO, LOr>) {
    return MPI_LOR;
  } else if constexpr (std::is_same_v<RO, LXor>) {
    return MPI_LXOR;
  } else if constexpr (std::is_same_v<RO, Max>) {
    return MPI_MAX;
  } else if constexpr (std::is_same_v<RO, MaxLoc>) {
    return MPI_MAXLOC;
  } else if constexpr (std::is_same_v<RO, Min>) {
    return MPI_MIN;
  } else if constexpr (std::is_same_v<RO, MinLoc>) {
    return MPI_MINLOC;
  } else if constexpr (std::is_same_v<RO, Sum>) {
    return MPI_SUM;
  } else if constexpr (std::is_same_v<RO, Prod>) {
    return MPI_PROD;
  } else {
    static_assert(std::is_void_v<RO>, "KokkosComm::Impl::mpi_reduction_op: operator not implemented");
    return MPI_SUM;  // unreachable
  }
}

#if defined(KOKKOSCOMM_ENABLE_NCCL)
template <ReductionOperator RO>
constexpr auto nccl_reduction_op() -> ncclRedOp_t {
  if constexpr (std::is_same_v<RO, Sum>) {
    return ncclSum;
  } else if constexpr (std::is_same_v<RO, Prod>) {
    return ncclProd;
  } else if constexpr (std::is_same_v<RO, Min>) {
    return ncclMin;
  } else if constexpr (std::is_same_v<RO, Max>) {
    return ncclMax;
  } else if constexpr (std::is_same_v<RO, Average>) {
    return ncclAvg;
  } else {
    static_assert(std::is_void_v<RO>, "KokkosComm::Impl::nccl_reduction_op: operator not implemented");
    return ncclSum;  // unreachable
  }
}
#endif

}  // namespace Impl

template <CommunicationSpace CS, ReductionOperator RO>
[[nodiscard]] constexpr auto reduction_op() -> typename CS::reduction_op_type {
  if constexpr (std::is_same_v<CS, MpiSpace>) {
    return Impl::mpi_reduction_op<RO>();
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  } else if constexpr (std::is_same_v<CS, Experimental::NcclSpace>) {
    return Impl::nccl_reduction_op<RO>();
#endif
  } else {
    static_assert(std::is_void_v<CS>,
                  "KokkosComm::reduction_op: conversion not implemented for this communication space");
    return Impl::mpi_reduction_op<RO>();  // unreachable
  }
}

template <CommunicationSpace CS, ReductionOperator RO>
[[nodiscard]] constexpr auto reduction_op_for(RO&&) -> typename CS::reduction_op_type {
  return reduction_op<CS, std::remove_cvref_t<RO>>();
}

}  // namespace KokkosComm
