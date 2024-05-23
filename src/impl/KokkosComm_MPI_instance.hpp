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
class Universe {
 private:
  Communicator<ExecSpace> _comm;
  MPI_Session _shandle;

 public:
  Universe(MPI_Session shandle, MPI_Comm comm)
      : _comm(Communicator<ExecSpace>::from_raw_unchecked(comm)), _shandle(shandle) {}

  ~Universe() {
    // FIXME: find out how to properly finalize the session
    // MPI_Session_finalize(&_shandle);
  }

  auto comm(void) -> Communicator<ExecSpace> { return _comm; }
};

template <KokkosExecutionSpace ExecSpace>
auto initialize(void) -> Universe<ExecSpace> {
  MPI_Info kokkoscomm_info = MPI_INFO_NULL;
  MPI_Info_create(&kokkoscomm_info);
  // TODO: error handling

  // Set threading level for our session
  constexpr char thrd_lvl_key[] = "thread_level";
  constexpr char thrd_lvl_val[] = "MPI_THREAD_MULTIPLE";
  MPI_Info_set(kokkoscomm_info, thrd_lvl_key, thrd_lvl_val);
  // TODO: error handling

#ifdef KOKKOSCOMM_CUDA_AWARE_MPI
  // Disable CUDA pointer attribute checks from MPI
  constexpr char cu_ptr_attr_key[] = "mpi_communication_pattern";
  constexpr char cu_ptr_attr_val[] = "MPI_CPU_TO_GPU";
  MPI_Info_set(kokkoscomm_info, cu_ptr_attr_key, cu_ptr_attr_val);
  // TODO: error handling
#endif

  MPI_Session kokkoscomm_shandle = MPI_SESSION_NULL;
  MPI_Session_init(kokkoscomm_info, MPI_ERRORS_RETURN, &kokkoscomm_shandle);
  // TODO: error handling

  MPI_Group kokkoscomm_group = MPI_GROUP_NULL;
  constexpr char pset_name[] = "mpi://WORLD";
  MPI_Group_from_session_pset(kokkoscomm_shandle, pset_name, &kokkoscomm_group);
  // TODO: error handling

  MPI_Comm kokkoscomm_comm = MPI_COMM_NULL;
  MPI_Comm_create_from_group(kokkoscomm_group, "kokkos-comm.default_session", MPI_INFO_NULL, MPI_ERRORS_RETURN,
                             &kokkoscomm_comm);
  // TODO: error handling

  // Resource release
  MPI_Group_free(&kokkoscomm_group);
  MPI_Info_free(&kokkoscomm_info);

  return Universe<ExecSpace>(kokkoscomm_shandle, kokkoscomm_comm);
}

template <KokkosExecutionSpace ExecSpace>
auto initialize(int &argc, char *argv[]) -> Universe<ExecSpace> {
  // Check that MPI was initiliazed and init if it wasn't
  int is_initialized;
  MPI_Initialized(&is_initialized);
  if (0 == is_initialized) {
    int required = MPI_THREAD_MULTIPLE, provided;
    MPI_Init_thread(&argc, &argv, required, &provided);
  }

  return initialize<ExecSpace>();
}

}  // namespace KokkosComm
