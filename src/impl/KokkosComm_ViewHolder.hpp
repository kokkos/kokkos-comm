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

#include "impl/KokkosComm_concepts.hpp"

namespace KokkosComm::Impl {

// a type-erased view. Request uses these to keep temporary views alive for
// the lifetime of "Immediate" MPI operations
struct ViewHolderBase {
  virtual ~ViewHolderBase() {}
};
template <KokkosView V>
struct ViewHolder : public ViewHolderBase, InvokableHolderBase {
  ViewHolder(const V &v) : v_(v) {}
  V v_;

  virtual void operator()() override { /* do nothing*/
  }
};

}  // namespace KokkosComm::Impl