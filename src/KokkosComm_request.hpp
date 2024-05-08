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

#include <memory>

#include "KokkosComm_request_impl.hpp"

namespace KokkosComm {

class Req {
 public:
  Req() : impl_(std::make_shared<Impl::Req>()) {}
  Req(const std::shared_ptr<Impl::Req> &impl) : impl_(impl) {}
  MPI_Request &mpi_req() { return impl_->mpi_req(); }
  void wait() { impl_->wait(); }

 private:
  std::shared_ptr<Impl::Req> impl_;
};

}  // namespace KokkosComm
