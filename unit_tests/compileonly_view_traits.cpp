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

///! \brief
/// make sure that an outside type can implement traits and be used with KokkosComm

#include <KokkosComm.hpp>



struct FakeView {};

namespace KokkosComm {

// make the FakeView implement the view interface
template <>
struct Traits<FakeView> {
  using pointer_type = char *;
  using scalar_type  = char;

  static bool is_contiguous(const FakeView &v) { return false; }

  static pointer_type data_handle(const FakeView &v) { return nullptr; }

  static size_t span(const FakeView &v) { return 0; }

  static size_t extent(const FakeView &v, const int i) { return 0; }

  static size_t stride(const FakeView &v, const int i) { return 0; }

  static constexpr bool is_reference_counted() { return false; }

  static constexpr size_t rank() { return 0; }
};

// Make the FakeView implement the pack interface
template <>
struct PackTraits<FakeView> {
  using packer_type = Impl::Packer::MpiDatatype<FakeView>;

  static bool needs_unpack(const FakeView &v) { return false; }
  static bool needs_pack(const FakeView &v) { return false; }
};

}  // namespace KokkosComm

// should compile
int main() {
  KokkosComm::isend(Kokkos::DefaultExecutionSpace(), FakeView(), 0, 0,
                    MPI_COMM_WORLD);
}