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
  struct operator {};                   \
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

template <ReductionOperator R>
constexpr auto mpi_reduction_op() -> MPI_Op {
  if constexpr (std::is_same_v<R, BAnd>) {
    return MPI_BAND;
  } else if constexpr (std::is_same_v<R, BOr>) {
    return MPI_BOR;
  } else if constexpr (std::is_same_v<R, BXor>) {
    return MPI_BXOR;
  } else if constexpr (std::is_same_v<R, LAnd>) {
    return MPI_LAND;
  } else if constexpr (std::is_same_v<R, LOr>) {
    return MPI_LOR;
  } else if constexpr (std::is_same_v<R, LXor>) {
    return MPI_LXOR;
  } else if constexpr (std::is_same_v<R, Max>) {
    return MPI_MAX;
  } else if constexpr (std::is_same_v<R, MaxLoc>) {
    return MPI_MAXLOC;
  } else if constexpr (std::is_same_v<R, Min>) {
    return MPI_MIN;
  } else if constexpr (std::is_same_v<R, MinLoc>) {
    return MPI_MINLOC;
  } else if constexpr (std::is_same_v<R, Sum>) {
    return MPI_SUM;
  } else if constexpr (std::is_same_v<R, Prod>) {
    return MPI_PROD;
  } else {
    static_assert(std::is_void_v<R>, "KokkosComm::Impl::mpi_reduction_op: operator not implemented");
    return MPI_SUM;  // unreachable
  }
}

#if defined(KOKKOSCOMM_ENABLE_NCCL)
template <ReductionOperator R>
constexpr auto nccl_reduction_op() -> ncclRedOp_t {
  if constexpr (std::is_same_v<R, Sum>) {
    return ncclSum;
  } else if constexpr (std::is_same_v<R, Prod>) {
    return ncclProd;
  } else if constexpr (std::is_same_v<R, Min>) {
    return ncclMin;
  } else if constexpr (std::is_same_v<R, Max>) {
    return ncclMax;
  } else if constexpr (std::is_same_v<R, Average>) {
    return ncclAvg;
  } else {
    static_assert(std::is_void_v<R>, "KokkosComm::Impl::nccl_reduction_op: operator not implemented");
    return ncclSum;  // unreachable
  }
}
#endif

}  // namespace Impl

/// @brief Converts a Kokkos Comm reduction operator tag to its communication space equivalent representation.
///
/// When `C` is:
/// - `MpiSpace`, returns the corresponding `MPI_Op` type.
/// - `NcclSpace`, returns the corresponding `ncclRedOp_t` type.
///
/// Non-system reduction operators (i.e. operators not natively supported by `C`) are not convertible.
///
/// @tparam C The target communication space backend to use for data type conversion.
/// @tparam R The Kokkos Comm reduction operator tag to convert.
/// @returns The communication space representation of the reduction operator.
template <CommunicationSpace C, ReductionOperator R>
[[nodiscard]] constexpr auto reduction_op() -> typename C::reduction_op_type {
  if constexpr (std::is_same_v<C, MpiSpace>) {
    return Impl::mpi_reduction_op<R>();
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  } else if constexpr (std::is_same_v<C, Experimental::NcclSpace>) {
    return Impl::nccl_reduction_op<R>();
#endif
  } else {
    static_assert(std::is_void_v<C>,
                  "KokkosComm::reduction_op: conversion not implemented for this communication space");
    return Impl::mpi_reduction_op<R>();  // unreachable
  }
}

template <CommunicationSpace C, ReductionOperator R>
[[nodiscard]] constexpr auto reduction_op_for([[maybe_unused]] R&& red_op) -> typename C::reduction_op_type {
  return reduction_op<C, std::remove_cvref_t<R>>();
}

template <CommunicationSpace C, ReductionOperator R>
[[nodiscard]] constexpr auto reduction_op_for([[maybe_unused]] C&& comm, [[maybe_unused]] R&& red_op) ->
    typename C::reduction_op_type {
  return reduction_op<C, std::remove_cvref_t<R>>();
}

}  // namespace KokkosComm
