#pragma once

#include <mpi.h>

#include <Kokkos_Core.hpp>

// impl
#include "KokkosComm_allocate.hpp"
#include "KokkosComm_pack.hpp"
#include "KokkosComm_types.hpp"
#include "KokkosComm_unpack.hpp"

template <typename View, typename ExecSpace>
auto pack_sv(const ExecSpace &space, const View &sv) {

  if (sv.span_is_contiguous()) {
    return sv;
  } else {
    using non_const_value_type = typename View::non_const_value_type;
    if constexpr (View::rank == 1) {
      Kokkos::View<non_const_value_type *> packed(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "packed"),
          sv.extent(0));
      pack(space, packed, sv);
      return packed;
    } else {
      static_assert(std::is_void_v<View>,
                    "KokkosComm::reduce only supports rank-1 send views");
    }
  }
}

namespace KokkosComm::Impl {
template <typename SendView, typename RecvView, typename ExecSpace>
void reduce(const ExecSpace &space, const SendView &sv, const RecvView &rv,
            MPI_Op op, int root, MPI_Comm comm) {
  using send_type = typename SendView::value_type;

  const int rank = [=]() -> int {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
  }();

  auto packed_sv = pack_sv(space, sv);
  auto packed_rv = (rank == root) ? allocate_packed("packed_rv", rv) : rv;

  MPI_Reduce(packed_sv.data(), packed_rv.data(),
             packed_sv.span() * sizeof(send_type), mpi_type_v<send_type>, op,
             root, comm);

  if (rank == root) {
    unpack(rv, packed_rv);
  }
}
} // namespace KokkosComm::Impl