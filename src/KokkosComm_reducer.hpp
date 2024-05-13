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
#include "impl/KokkosComm_include_mpi.hpp"

namespace KokkosComm {

class Reducer {
 private:
  MPI_Op _op;

 public:
  Reducer(MPI_Op op) : _op{op} {}
  operator MPI_Op() const { return _op; }
};

}  // namespace KokkosComm
