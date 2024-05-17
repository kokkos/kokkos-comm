#pragma once

#include "impl/KokkosComm_concepts.hpp"
#include "impl/KokkosComm_NonContigCtx.hpp"
#include "impl/KokkosComm_types.hpp"
#include "impl/KokkosComm_request.hpp"

// use deep-copy to make non-contiguous views a single contiguous block

namespace KokkosComm::Impl {

template <KokkosExecutionSpace Space, KokkosView SendView, KokkosView RecvView>
struct NonContigDeepCopyReduce {
  using ctx_type = CtxReduce;
  using ST       = KokkosComm::Traits<SendView>;
  using RT       = KokkosComm::Traits<RecvView>;

  static_assert(std::is_same_v<typename SendView::non_const_value_type, typename RecvView::non_const_value_type>,
                "");  // FIXME: message
  using Scalar = typename SendView::non_const_value_type;

  static ctx_type pre_reduce(const Space &space, const SendView &sv, const RecvView &rv, int count) {
    ctx_type ctx;

    const bool s_contig = ST::is_contiguous(sv);
    const bool r_contig = RT::is_contiguous(rv);

    if (s_contig && r_contig) {
      // FIXME: is asynchronous
      ctx.mpi_args.push_back(ctx_type::MpiArgs(ST::data_handle(sv), RT::data_handle(rv), mpi_type_v<Scalar>, count));
    } else if (!s_contig && r_contig) {
      // queue allocation of contiguous buffer
      auto packed = allocate_for("FIXME", space, sv);  // FIXME: need to fence if pointer isn't valid immediately
      // keep buffer alive during operation
      ctx.views.push_back(std::make_shared<ViewHolder<decltype(packed)>>(packed));
      // queue copy of data to buffer
      Kokkos::deep_copy(space, packed, sv);
      ctx.mpi_args.push_back(ctx_type::MpiArgs(KokkosComm::Traits<decltype(packed)>::data_handle(packed),
                                               RT::data_handle(rv), mpi_type_v<Scalar>, count));
      ctx.set_pre_uses_space();
    } else if (s_contig && !r_contig) {
      auto packed = allocate_for("FIXME", space, rv);  // FIXME: need to fence if pointer isn't valid immediately
      ctx.views.push_back(std::make_shared<ViewHolder<decltype(packed)>>(packed));
      Kokkos::deep_copy(space, packed, rv);
      ctx.mpi_args.push_back(ctx_type::MpiArgs(
          ST::data_handle(sv), KokkosComm::Traits<decltype(packed)>::data_handle(packed), mpi_type_v<Scalar>, count));
      ctx.set_pre_uses_space();
    } else {
      auto spacked = allocate_for("FIXME", space, sv);
      ctx.views.push_back(std::make_shared<ViewHolder<decltype(spacked)>>(spacked));
      Kokkos::deep_copy(space, spacked, sv);
      auto rpacked = allocate_for("FIXME", space, rv);  // FIXME: need to fence if pointer isn't valid immediately
      ctx.views.push_back(std::make_shared<ViewHolder<decltype(rpacked)>>(rpacked));
      Kokkos::deep_copy(space, rpacked, rv);
      ctx.mpi_args.push_back(ctx_type::MpiArgs(KokkosComm::Traits<decltype(spacked)>::data_handle(spacked),
                                               KokkosComm::Traits<decltype(rpacked)>::data_handle(rpacked),
                                               mpi_type_v<Scalar>, count));
      ctx.set_pre_uses_space();
    }

    return ctx;
  }

  static CtxReduce &post_reduce(const Space &space, const RecvView &rv, CtxReduce &ctx) {
    if (!RT::is_contiguous(rv)) {
      for (ctx_type::MpiArgs &args : ctx.mpi_args) {
        using UVT = Kokkos::View<Scalar *, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged>;

        if constexpr (RT::rank() == 1) {
          UVT ub(static_cast<Scalar *>(args.rbuf), rv.extent(0));
          Kokkos::deep_copy(space, rv, ub);
          ctx.set_post_uses_space();
        } else if constexpr (RT::rank() == 2) {
          UVT ub(static_cast<Scalar *>(args.rbuf), rv.extent(0), rv.extent(1));
          Kokkos::deep_copy(space, rv, ub);
          ctx.set_post_uses_space();
        } else {
          static_assert(std::is_void_v<RecvView>, "FIXME");  // FIXME: message
        }
      }
    }
    return ctx;
  }

 private:
  template <KokkosView View>
  static auto allocate_for(const std::string &label, const Space &space, const View &view) {
    using non_const_packed_view_type =
        Kokkos::View<typename View::non_const_data_type, Kokkos::LayoutRight, typename View::memory_space>;
    using PVT = KokkosComm::Traits<non_const_packed_view_type>;

    if constexpr (KokkosComm::Traits<View>::rank() == 1) {
      return non_const_packed_view_type(Kokkos::view_alloc(space, Kokkos::WithoutInitializing, label), view.extent(0));
    } else if constexpr (KokkosComm::Traits<View>::rank() == 2) {
      return non_const_packed_view_type(Kokkos::view_alloc(space, Kokkos::WithoutInitializing, label), view.extent(0),
                                        view.extent(1));
    } else {
      static_assert(std::is_void_v<View>, "NonContigDeepCopySendRecv for rank >= 2 views unimplemented");
    }
  }
};

};  // namespace KokkosComm::Impl
