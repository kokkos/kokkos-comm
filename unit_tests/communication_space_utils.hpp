// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#if defined(KOKKOSCOMM_ENABLE_MPI)
#include <mpi.h>
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL) || defined(KOKKOSCOMM_ENABLE_RCCL)
#include "nccl/utils.hpp"
#endif

namespace test_utils {

template <typename>
inline constexpr bool dependent_false_v = false;

template <KokkosComm::CommunicationSpace CommSpace>
auto get_raw_comm() -> typename CommSpace::communicator_type {
#if defined(KOKKOSCOMM_ENABLE_MPI)
  if constexpr (std::is_same_v<CommSpace, KokkosComm::MpiSpace>) {
    return MPI_COMM_WORLD;
  } else
#endif
#if defined(KOKKOSCOMM_ENABLE_NCCL)
      if constexpr (std::is_same_v<CommSpace, KokkosComm::Experimental::NcclSpace>) {
    return XcclCtx::get().comm();
  } else
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
  if constexpr (std::is_same_v<CommSpace, KokkosComm::Experimental::RcclSpace>) {
    return XcclCtx::get().comm();
  } else
#endif
  {
    static_assert(dependent_false_v<CommSpace>, "No raw communicator available for this communication space");
  }
}

template <KokkosComm::CommunicationSpace CommSpace>
auto get_exec() -> Kokkos::DefaultExecutionSpace {
#if defined(KOKKOSCOMM_ENABLE_MPI)
  if constexpr (std::is_same_v<CommSpace, KokkosComm::MpiSpace>) {
    return Kokkos::DefaultExecutionSpace{};
  } else
#endif
#if defined(KOKKOSCOMM_ENABLE_NCCL)
      if constexpr (std::is_same_v<CommSpace, KokkosComm::Experimental::NcclSpace>) {
    return Kokkos::DefaultExecutionSpace{XcclCtx::get().stream()};
  } else
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
  if constexpr (std::is_same_v<CommSpace, KokkosComm::Experimental::RcclSpace>) {
    return Kokkos::DefaultExecutionSpace{XcclCtx::get().stream()};
  } else
#endif
  {
    static_assert(dependent_false_v<CommSpace>, "No execution-space instance available for this communication space");
  }
}

template <KokkosComm::CommunicationSpace CommSpace>
auto make_communicator() -> KokkosComm::Communicator<CommSpace> {
  return KokkosComm::Communicator<CommSpace>::from_raw(get_raw_comm<CommSpace>(), get_exec<CommSpace>());
}

template <KokkosComm::CommunicationSpace CommSpace>
auto get_comm_size(typename CommSpace::communicator_type comm) -> int {
#if defined(KOKKOSCOMM_ENABLE_MPI)
  if constexpr (std::is_same_v<CommSpace, KokkosComm::MpiSpace>) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
  } else
#endif
#if defined(KOKKOSCOMM_ENABLE_NCCL) || defined(KOKKOSCOMM_ENABLE_RCCL)
#if defined(KOKKOSCOMM_ENABLE_NCCL)
      if constexpr (std::is_same_v<CommSpace, KokkosComm::Experimental::NcclSpace>) {
#else
      if constexpr (std::is_same_v<CommSpace, KokkosComm::Experimental::RcclSpace>) {
#endif
    int size;
    ncclCommCount(comm, &size);
    return size;
  } else
#endif
  {
    static_assert(dependent_false_v<CommSpace>, "No size query available for this communication space");
  }
}

template <KokkosComm::CommunicationSpace CommSpace>
auto get_comm_rank(typename CommSpace::communicator_type comm) -> int {
#if defined(KOKKOSCOMM_ENABLE_MPI)
  if constexpr (std::is_same_v<CommSpace, KokkosComm::MpiSpace>) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
  } else
#endif
#if defined(KOKKOSCOMM_ENABLE_NCCL) || defined(KOKKOSCOMM_ENABLE_RCCL)
#if defined(KOKKOSCOMM_ENABLE_NCCL)
      if constexpr (std::is_same_v<CommSpace, KokkosComm::Experimental::NcclSpace>) {
#else
      if constexpr (std::is_same_v<CommSpace, KokkosComm::Experimental::RcclSpace>) {
#endif
    int rank;
    ncclCommUserRank(comm, &rank);
    return rank;
  } else
#endif
  {
    static_assert(dependent_false_v<CommSpace>, "No rank query available for this communication space");
  }
}

template <KokkosComm::CommunicationSpace CommSpace>
auto get_undefined_color() -> int {
#if defined(KOKKOSCOMM_ENABLE_MPI)
  if constexpr (std::is_same_v<CommSpace, KokkosComm::MpiSpace>) {
    return MPI_UNDEFINED;
  } else
#endif
#if defined(KOKKOSCOMM_ENABLE_NCCL) || defined(KOKKOSCOMM_ENABLE_RCCL)
  {
    return NCCL_SPLIT_NOCOLOR;
  }
#else
  { static_assert(dependent_false_v<CommSpace>, "No undefined color available for this communication space"); }
#endif
}

}  // namespace test_utils
