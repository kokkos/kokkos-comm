#pragma once

#include "impl/KokkosComm_concepts.hpp"
#include "impl/KokkosComm_NonContigCtx.hpp"
#include "impl/KokkosComm_types.hpp"
#include "impl/KokkosComm_request.hpp"
#include "impl/KokkosComm_NonContig_Base.hpp"

// constructs an MPI datatype for non-contiguous data

namespace KokkosComm::Impl {

template <Impl::KokkosExecutionSpace Space, Impl::KokkosView View>
struct NonContigDatatypes : public NonContigBase<NonContigDatatypes> {
  using KCT                 = KokkosComm::Traits<View>;
  using non_const_data_type = typename View::non_const_data_type;

  using non_const_packed_view_type =
      Kokkos::View<non_const_data_type, Kokkos::LayoutRight, typename View::memory_space>;
  using PVT = KokkosComm::Traits<non_const_packed_view_type>;

  using value_type = typename View::non_const_value_type;

  static Ctx pre_send(const Space &space, const View &view, const std::vector<int> &counts, const std::vector<int> &displs) {
    Ctx ctx;

    MPI_Datatype type = mpi_type<value_type>();
    int count         = 1;
    if (KCT::rank() < 2 && KCT::is_contiguous(view)) {
      count = KCT::extent(view, 0);
    } else {
      // This doesn't work for 1D contiguous views into reduce because it
      // represents the whole 1D view as 1 Hvector, rather than N elements.
      // FIXME: is there a more generic way to handle this, maybe by treating
      // the last dimension specially under certain circumstances?
      for (size_t d = 0; d < KCT::rank(); ++d) {
        MPI_Datatype newtype;
        MPI_Type_create_hvector(KCT::extent(view, d) /*count*/, 1 /*block length*/,
                                KCT::stride(view, d) * sizeof(value_type), type, &newtype);
        type = newtype;
      }
    }
    MPI_Type_commit(&type);
    ctx.mpi_args.push_back(Ctx::MpiArgs(KCT::data_handle(view), type, count));

    return ctx;
  }

  static Ctx pre_recv(const Space &space, const View &view, const std::vector<int> &counts, const std::vector<int> &displs) {
    Ctx ctx;

    using KCT = KokkosComm::Traits<View>;

    MPI_Datatype type = mpi_type<value_type>();
    int count         = 1;
    if (KCT::rank() < 2 && KCT::is_contiguous(view)) {
      count = KCT::extent(view, 0);
    } else {
      // This doesn't work for 1D contiguous views into reduce because it
      // represents the whole 1D view as 1 Hvector, rather than N elements.
      // FIXME: is there a more generic way to handle this, maybe by treating
      // the last dimension specially under certain circumstances?
      for (size_t d = 0; d < KCT::rank(); ++d) {
        MPI_Datatype newtype;
        MPI_Type_create_hvector(KCT::extent(view, d) /*count*/, 1 /*block length*/,
                                KCT::stride(view, d) * sizeof(value_type), type, &newtype);
        type = newtype;
      }
    }
    MPI_Type_commit(&type);
    ctx.mpi_args.push_back(Ctx::MpiArgs(KCT::data_handle(view), type, count));

    return ctx;
  }

  static Ctx post_recv(const Space & /*space*/, const View & /*view*/, Ctx &ctx) {
    return ctx;  // nothing to do
  }
};

};  // namespace KokkosComm::Impl
