// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>
#include "packer.hpp"

namespace KokkosComm::mpi::Impl {

template <typename T>
struct PackTraits {
  static_assert(std::is_void_v<T>, "KokkosComm::PackTraits not specialized for requested type");
};

/*! \brief This can be specialized to do custom behavior for a particular view*/
template <KokkosView View>
struct PackTraits<View> {
  using packer_type = Packer::DeepCopy<View>;
};

}  // namespace KokkosComm::mpi::Impl
