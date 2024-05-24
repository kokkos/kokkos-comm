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

#include "KokkosComm_communicator.hpp"
#include "KokkosComm_concepts.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

namespace KokkosComm {

template <KokkosExecutionSpace ExecSpace>
class Context {
 private:
  MPI_Session _shandle;
  Communicator<ExecSpace> _comm;

 public:
  Context(MPI_Session shandle, MPI_Comm comm) : _shandle(shandle), _comm(Communicator<ExecSpace>(comm)) {}

  ~Context() {
    // Ensure the session-associated communicator is destroyed before the session is finalized.
    _comm.~Communicator();
    MPI_Session_finalize(&_shandle);
  }

  auto comm(void) -> const Communicator<ExecSpace>& { return _comm; }
};

template <KokkosExecutionSpace ExecSpace>
auto initialize(void) -> Context<ExecSpace> {
  int rc;

  MPI_Session kokkoscomm_shandle = MPI_SESSION_NULL;
  MPI_Group kokkoscomm_group     = MPI_GROUP_NULL;
  MPI_Comm kokkoscomm_comm       = MPI_COMM_NULL;
  MPI_Info kokkoscomm_info       = MPI_INFO_NULL;

  MPI_Info_create(&kokkoscomm_info);

  // Set threading level for our session
  constexpr char thrd_lvl_key[] = "thread_level";
  constexpr char thrd_lvl_val[] = "MPI_THREAD_MULTIPLE";
  MPI_Info_set(kokkoscomm_info, thrd_lvl_key, thrd_lvl_val);

#ifdef KOKKOSCOMM_CUDA_AWARE_MPI
  // Disable CUDA pointer attribute checks from MPI
  constexpr char cu_ptr_attr_key[] = "mpi_communication_pattern";
  constexpr char cu_ptr_attr_val[] = "MPI_CPU_TO_GPU";
  MPI_Info_set(kokkoscomm_info, cu_ptr_attr_key, cu_ptr_attr_val);
#endif

  rc = MPI_Session_init(kokkoscomm_info, MPI_ERRORS_RETURN, &kokkoscomm_shandle);

  constexpr char pset_name[] = "mpi://WORLD";
  MPI_Group_from_session_pset(kokkoscomm_shandle, pset_name, &kokkoscomm_group);

  MPI_Comm_create_from_group(kokkoscomm_group, "kokkos-comm.default_session", MPI_INFO_NULL, MPI_ERRORS_RETURN,
                             &kokkoscomm_comm);

  return Context<ExecSpace>(kokkoscomm_shandle, kokkoscomm_comm);
}

}  // namespace KokkosComm
