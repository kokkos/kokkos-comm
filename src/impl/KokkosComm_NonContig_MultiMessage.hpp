#pragma once

#include "impl/KokkosComm_concepts.hpp"
#include "impl/KokkosComm_NonContigCtx.hpp"
#include "impl/KokkosComm_types.hpp"
#include "impl/KokkosComm_request.hpp"

// sends a single message for each word in the view

namespace KokkosComm::Impl {

template <Impl::KokkosExecutionSpace Space, Impl::KokkosView View>
struct NonContigMultiMessage {
  using KCT                 = KokkosComm::Traits<View>;
  using non_const_data_type = typename View::non_const_data_type;

  using non_const_packed_view_type =
      Kokkos::View<non_const_data_type, Kokkos::LayoutRight, typename View::memory_space>;
  using PVT = KokkosComm::Traits<non_const_packed_view_type>;

  static Ctx pre_send(const Space &space, const View &view, int sendCount) {
    using KCT = KokkosComm::Traits<View>;
    Ctx ctx;

    if constexpr (KCT::rank() == 1) {
      for (size_t i = 0; i < view.extent(0); ++i) {
        ctx.mpi_args.push_back(
            Ctx::MpiArgs(&view[i], mpi_type<typename non_const_packed_view_type::non_const_value_type>(), 1));
      }
    } else if constexpr (KCT::rank() == 2) {
      for (size_t i = 0; i < view.extent(0); ++i) {
        for (size_t j = 0; j < view.extent(1); ++j) {
          ctx.mpi_args.push_back(
              Ctx::MpiArgs(&view(i, j), mpi_type<typename non_const_packed_view_type::non_const_value_type>(), 1));
        }
      }
    } else {
      static_assert(std::is_void_v<View>, "unsupported");
    }

    return ctx;
  }

  static Ctx pre_send(const Space &space, const View &view) {
    
    return pre_send(space, view, KokkosComm::Traits<View>::span(view));

  }

  static Ctx pre_recv(const Space &space, const View &view) {
    Ctx ctx;

    using KCT = KokkosComm::Traits<View>;

    if constexpr (KCT::rank() == 1) {
      for (size_t i = 0; i < view.extent(0); ++i) {
        ctx.mpi_args.push_back(
            Ctx::MpiArgs(&view[i], mpi_type<typename non_const_packed_view_type::non_const_value_type>(), 1));
      }
    } else if constexpr (KCT::rank() == 2) {
      for (size_t i = 0; i < view.extent(0); ++i) {
        for (size_t j = 0; j < view.extent(1); ++j) {
          ctx.mpi_args.push_back(
              Ctx::MpiArgs(&view(i, j), mpi_type<typename non_const_packed_view_type::non_const_value_type>(), 1));
        }
      }
    } else {
      static_assert(std::is_void_v<View>, "unsupported");
    }

    return ctx;
  }

  static Ctx post_recv(const Space & /*space*/, const View & /*view*/, Ctx &ctx) {
    return ctx;  // nothing to do
  }
};

};  // namespace KokkosComm::Impl
