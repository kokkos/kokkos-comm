#pragma once

#include "impl/KokkosComm_concepts.hpp"
#include "impl/KokkosComm_NonContigCtx.hpp"
#include "impl/KokkosComm_types.hpp"
#include "impl/KokkosComm_request.hpp"

// use deep-copy to make non-contiguous views a single contiguous block

namespace KokkosComm::Impl {

template <KokkosExecutionSpace Space, KokkosView View>
struct NonContigDeepCopySendRecv {
  using KCT                 = KokkosComm::Traits<View>;
  using non_const_data_type = typename View::non_const_data_type;

  using non_const_packed_view_type =
      Kokkos::View<non_const_data_type, Kokkos::LayoutRight, typename View::memory_space>;
  using PVT = KokkosComm::Traits<non_const_packed_view_type>;

  static non_const_packed_view_type allocate_for(const Space &space, const View &view) {
    const std::string label = "";  // FIXME

    if constexpr (KCT::rank() == 1) {
      return non_const_packed_view_type(Kokkos::view_alloc(space, Kokkos::WithoutInitializing, label), view.extent(0));
    } else if constexpr (KCT::rank() == 2) {
      return non_const_packed_view_type(Kokkos::view_alloc(space, Kokkos::WithoutInitializing, label), view.extent(0),
                                        view.extent(1));
    } else {
      static_assert(std::is_void_v<View>, "NonContigDeepCopySendRecv for rank >= 2 views unimplemented");
    }
  }



  static CtxBufCount pre_send(const Space &space, const View &view) {

    CtxBufCount ctx;

    using KCT = KokkosComm::Traits<View>;

    if (KCT::is_contiguous(view)) {
      ctx.mpi_args.push_back(
          CtxBufCount::MpiArgs(KCT::data_handle(view), mpi_type<typename View::non_const_value_type>(), KCT::span(view)));
      ctx.wait_callbacks.push_back(std::make_shared<ViewHolder<View>>(view));
    } else {
      non_const_packed_view_type packed = allocate_for(space, view);
      Kokkos::deep_copy(space, packed, view);
      ctx.set_pre_uses_space();
      ctx.mpi_args.push_back(CtxBufCount::MpiArgs(PVT::data_handle(packed),
                                          mpi_type<typename non_const_packed_view_type::non_const_value_type>(),
                                          packed.size()));
      ctx.wait_callbacks.push_back(std::make_shared<ViewHolder<non_const_packed_view_type>>(packed));
    }

    return ctx;
  }

  static CtxBufCount pre_recv(const Space &space, const View &view) {
    CtxBufCount ctx;

    if (KCT::is_contiguous(view)) {
      ctx.mpi_args.push_back(
          CtxBufCount::MpiArgs(KCT::data_handle(view), mpi_type<typename View::non_const_value_type>(), KCT::span(view)));
    } else {
      non_const_packed_view_type packed = allocate_for(space, view);
      Kokkos::deep_copy(space, packed, view);
      ctx.set_pre_uses_space();
      ctx.mpi_args.push_back(CtxBufCount::MpiArgs(PVT::data_handle(packed),
                                          mpi_type<typename non_const_packed_view_type::non_const_value_type>(),
                                          packed.size()));
      ctx.views.push_back(std::make_shared<ViewHolder<non_const_packed_view_type>>(packed));
    }
    return ctx;
  }

  static CtxBufCount post_recv(const Space &space, const View &view, CtxBufCount &ctx) {
    if (!KCT::is_contiguous(view)) {
      for (CtxBufCount::MpiArgs &args : ctx.mpi_args) {
        using UVT = Kokkos::View<non_const_data_type, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged>;

        if constexpr (KCT::rank() == 1) {
          UVT ub(static_cast<View::non_const_value_type *>(args.buf), view.extent(0));
          Kokkos::deep_copy(space, view, ub);
          ctx.set_post_uses_space();
        } else if constexpr (KCT::rank() == 2) {
          UVT ub(static_cast<View::non_const_value_type *>(args.buf), view.extent(0), view.extent(1));
          Kokkos::deep_copy(space, view, ub);
          ctx.set_post_uses_space();
        } else {
          static_assert(std::is_void_v<View>, "FIXME");
        }
      }
    }
    return ctx;
  }

};

};  // namespace KokkosComm::Impl
