// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>

#include "impl/types.hpp"

namespace KokkosComm::mpi {

template <KokkosView SendView, KokkosView RecvView>
void inclusive_scan(SendView const &sv, RecvView const &rv, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::inclusive_scan");

  using SendScalar = typename SendView::value_type;
  using RecvScalar = typename RecvView::value_type;
  static_assert(std::is_same_v<std::remove_cv_t<SendScalar>, std::remove_cv_t<RecvScalar> >,
                "Send and receive views have different value types");

  static_assert(KokkosComm::rank<SendView>() <= 1, "inclusive_scan for SendView::rank > 1 not supported");
  static_assert(KokkosComm::rank<RecvView>() <= 1, "inclusive_scan for RecvView::rank > 1 not supported");

  if (!KokkosComm::is_contiguous(sv)) {
    throw std::runtime_error{"low-level inclusive_scan requires contiguous send view"};
  }
  if (!KokkosComm::is_contiguous(rv)) {
    throw std::runtime_error{"low-level inclusive_scan requires contiguous recv view"};
  }
  if (sv.size() != rv.size()) {
    throw std::runtime_error{"inclusive_scan requires send and receive views to have the same size"};
  }
  int const count = sv.size();
  MPI_Scan(KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), count, KokkosComm::Impl::mpi_type_v<SendScalar>,
           op, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosView SendView, KokkosView RecvView>
void exclusive_scan(SendView const &sv, RecvView const &rv, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::exclusive_scan");

  using SendScalar = typename SendView::value_type;
  using RecvScalar = typename RecvView::value_type;
  static_assert(std::is_same_v<std::remove_cv_t<SendScalar>, std::remove_cv_t<RecvScalar> >,
                "Send and receive views have different value types");

  static_assert(KokkosComm::rank<SendView>() <= 1, "exclusive_scan for SendView::rank > 1 not supported");
  static_assert(KokkosComm::rank<RecvView>() <= 1, "exclusive_scan for RecvView::rank > 1 not supported");

  if (!KokkosComm::is_contiguous(sv)) {
    throw std::runtime_error{"low-level exclusive_scan requires contiguous send view"};
  }
  if (!KokkosComm::is_contiguous(rv)) {
    throw std::runtime_error{"low-level exclusive_scan requires contiguous recv view"};
  }
  if (sv.size() != rv.size()) {
    throw std::runtime_error{"exclusive_scan requires send and receive views to have the same size"};
  }
  int const count = sv.size();
  MPI_Exscan(KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), count, KokkosComm::Impl::mpi_type_v<SendScalar>,
             op, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void inclusive_scan(ExecSpace const &space, SendView const &sv, RecvView const &rv, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::inclusive_scan");

  if (!KokkosComm::is_contiguous(sv) || !KokkosComm::is_contiguous(rv)) {
    throw std::runtime_error("inclusive_scan for non-contiguous views not implemented");
  }
  space.fence("fence before inclusive_scan");  // work in space may have been used to produce send view data
  inclusive_scan(sv, rv, op, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void exclusive_scan(ExecSpace const &space, SendView const &sv, RecvView const &rv, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::exclusive_scan");

  if (!KokkosComm::is_contiguous(sv) || !KokkosComm::is_contiguous(rv)) {
    throw std::runtime_error("exclusive_scan for non-contiguous views not implemented");
  }
  space.fence("fence before exclusive_scan");  // work in space may have been used to produce send view data
  exclusive_scan(sv, rv, op, comm);

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::mpi
