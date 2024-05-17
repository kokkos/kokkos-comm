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

#include <Kokkos_Core.hpp>

#include "KokkosComm_pack_traits.hpp"
#include "KokkosComm_traits.hpp"

// impl
#include "KokkosComm_include_mpi.hpp"
#include "KokkosComm_types.hpp"

namespace KokkosComm::Impl {
template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void alltoall(const ExecSpace &space, const SendView &sv, const size_t sendCount, const RecvView &rv,
              const size_t recvCount, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::alltoall");

  using ST         = KokkosComm::Traits<SendView>;
  using RT         = KokkosComm::Traits<RecvView>;
  using SendScalar = typename SendView::value_type;
  using RecvScalar = typename RecvView::value_type;

  static_assert(ST::rank() <= 1, "alltoall for SendView::rank > 1 not supported");
  static_assert(RT::rank() <= 1, "alltoall for RecvView::rank > 1 not supported");

  if (KokkosComm::PackTraits<SendView>::needs_pack(sv) || KokkosComm::PackTraits<RecvView>::needs_pack(rv)) {
    throw std::runtime_error("alltoall for non-contiguous views not implemented");
  } else {
    int size;
    MPI_Comm_size(comm, &size);

    if (sendCount * size > ST::extent(sv, 0)) {
      std::stringstream ss;
      ss << "alltoall sendCount * communicator size (" << sendCount << " * " << size
         << ") is greater than send view size";
      throw std::runtime_error(ss.str());
    }
    if (recvCount * size > RT::extent(rv, 0)) {
      std::stringstream ss;
      ss << "alltoall recvCount * communicator size (" << recvCount << " * " << size
         << ") is greater than recv view size";
      throw std::runtime_error(ss.str());
    }

    MPI_Alltoall(ST::data_handle(sv), sendCount, mpi_type_v<SendScalar>, RT::data_handle(rv), recvCount,
                 mpi_type_v<RecvScalar>, comm);
  }

  Kokkos::Tools::popRegion();
}

// in-place alltoall
template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void alltoall(const ExecSpace &space, const RecvView &rv, const size_t recvCount, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::Impl::alltoall");

  using RT         = KokkosComm::Traits<RecvView>;
  using RecvScalar = typename RecvView::value_type;

  static_assert(RT::rank() <= 1, "alltoall for RecvView::rank > 1 not supported");

  if (KokkosComm::PackTraits<RecvView>::needs_pack(rv)) {
    throw std::runtime_error("alltoall for non-contiguous views not implemented");
  } else {
    int size;
    MPI_Comm_size(comm, &size);

    if (recvCount * size > RT::extent(rv, 0)) {
      std::stringstream ss;
      ss << "alltoall recvCount * communicator size (" << recvCount << " * " << size
         << ") is greater than recv view size";
      throw std::runtime_error(ss.str());
    }

    MPI_Alltoall(MPI_IN_PLACE, 0 /*ignored*/, MPI_BYTE /*ignored*/, RT::data_handle(rv), recvCount,
                 mpi_type_v<RecvScalar>, comm);
  }

  Kokkos::Tools::popRegion();
}

}  // namespace KokkosComm::Impl
