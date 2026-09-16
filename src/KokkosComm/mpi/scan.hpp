// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <type_traits>

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "mpi_space.hpp"

namespace KokkosComm::mpi {

template <KokkosView SendView, MutKokkosView RecvView>
void inclusive_scan(SendView const &sv, RecvView const &rv, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::inclusive_scan");

  using SendScalar = typename SendView::non_const_value_type;
  using RecvScalar = typename RecvView::non_const_value_type;
  static_assert(
      std::is_same_v<SendScalar, RecvScalar>, "KokkosComm::mpi::inclusive_scan: View value types must be identical"
  );

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
  MPI_Scan(KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), count, datatype<MpiSpace, SendScalar>(), op, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosView SendView, MutKokkosView RecvView>
void exclusive_scan(SendView const &sv, RecvView const &rv, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::exclusive_scan");

  using SendScalar = typename SendView::non_const_value_type;
  using RecvScalar = typename RecvView::non_const_value_type;
  static_assert(
      std::is_same_v<SendScalar, RecvScalar>, "KokkosComm::mpi::inclusive_scan: View value types must be identical"
  );
  // FIXME_EXTERNAL #204, #226
#if defined(KOKKOSCOMM_IMPL_MPI_IS_MPICH) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
  // Unsupported if running MPICH and Views are in CUDA or HIP execution spaces
  // And that their value type is one of: `double`, `complex<float>`, `complex<double>`
  if constexpr (std::is_same_v<SendScalar, double> or std::is_same_v<SendScalar, Kokkos::complex<float>> or std::is_same_v<SendScalar, Kokkos::complex<double>>) {
    static_assert(
#if defined(KOKKOS_ENABLE_CUDA)
        not std::is_same_v<typename SendView::execution_space, Kokkos::Cuda> and
            not std::is_same_v<typename RecvView::execution_space, Kokkos::Cuda>,
#elif defined(KOKKOS_ENABLE_HIP)
        not std::is_same_v<typename SendView::execution_space, Kokkos::HIP> and
            not std::is_same_v<typename RecvView::execution_space, Kokkos::HIP>,
#endif
        "KokkosComm::mpi::exclusive_scan: Unsupported with MPICH + Kokkos CUDA/HIP backend"
    );
  }
#endif

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
  MPI_Exscan(
      KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), count, datatype<MpiSpace, SendScalar>(), op, comm
  );

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, MutKokkosView RecvView>
void inclusive_scan(ExecSpace const &space, SendView const &sv, RecvView const &rv, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::inclusive_scan");

  if (!KokkosComm::is_contiguous(sv) || !KokkosComm::is_contiguous(rv)) {
    throw std::runtime_error("inclusive_scan for non-contiguous views not implemented");
  }
  space.fence("fence before inclusive_scan");  // work in space may have been used to produce send view data
  inclusive_scan(sv, rv, op, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, MutKokkosView RecvView>
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
