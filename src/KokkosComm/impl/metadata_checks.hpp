#pragma once

#include <Kokkos_Core.hpp>

#include <utility>

#include <KokkosComm/traits.hpp>

bool constexpr metadata_checks =
#ifdef KOKKOSCOMM_ENABLE_DEBUG_RUNTIME_METADATA_CHECKS
    true;
#else
    false;
#endif

namespace KokkosComm::Impl {
namespace checks {

// Static (compile-time) checks

template <KokkosView SendView, MutKokkosView RecvView>
void static_assert_dtype_match(const SendView&, const RecvView&) {
  using ST = typename SendView::non_const_value_type;
  using RT = typename RecvView::non_const_value_type;
  static_assert(std::is_same_v<ST, RT>, "Send and Recv Views value types must have the same non-const type");
}

template <KokkosView SendView, MutKokkosView RecvView>
void static_assert_rank_match(const SendView&, const RecvView&) {
  static_assert(SendView::rank() == RecvView::rank(), "Send and Recv Views must have the same rank");
}

template <MutKokkosView View>
void static_assert_rank_leq_1(const View&) {
  static_assert(rank<View>() <= 1, "Views with rank higher than 1 are not supported");
}

// Checks that Send and Recv View ranks are compatible for allgather
// Recv may carry one extra leading dimension to index the contributing rank.
template <KokkosView SendView, MutKokkosView RecvView>
void static_assert_rank_match_allgather(const SendView&, const RecvView&) {
  static_assert(
      (RecvView::rank() == SendView::rank()) || (RecvView::rank() == SendView::rank() + 1),
      "Recv View must have the same rank as the Send View, or exactly one more"
  );
}

// Runtime checks

template <KokkosView SendView, MutKokkosView RecvView>
void fail_if_extents_mismatch(const SendView& sv, const RecvView& rv, const char* fn) {
  if constexpr (metadata_checks) {
    for (std::size_t i = 0; i < RecvView::rank(); ++i) {
      if (KokkosComm::extent(sv, i) != KokkosComm::extent(rv, i)) {
        std::stringstream ss;
        ss << fn << ": extent mismatch at dimension " << i << ": send view has " << KokkosComm::extent(sv, i)
           << ", recv view has " << KokkosComm::extent(rv, i);
        KokkosComm::nccl::fail_if(true, ss.str().c_str());
      }
    }
  }
}

// Checks that Send and Recv View sizes are compatible for allgather
// Recv has to be comm_size larger than Send
template <KokkosView SendView, MutKokkosView RecvView>
void fail_if_size_mismatch_allgather(const SendView& sv, const RecvView& rv, const char* fn, const int comm_size) {
  if constexpr (metadata_checks) {
    if (sv.size() * comm_size != rv.size()) {
      std::stringstream ss;
      ss << fn << ": size mismatch for allgather: send view has " << sv.size() << ", recv view has " << rv.size();
      ss << ", sv.size()*comm_size == rv.size() should hold";
      KokkosComm::nccl::fail_if(true, ss.str().c_str());
    }
  }
}

template <typename View>
void fail_if_noncontiguous(const View& v, const char* fn) {
  if (!is_contiguous(v)) {
    std::stringstream ss;
    ss << fn << ": unimplemented for non-contiguous Views: \"" << v.label() << "\" is non-contiguous";
    KokkosComm::nccl::fail_if(true, ss.str().c_str());
  }
}

// AllToAll specific: check count vs view size consistency
template <KokkosView View>
void fail_if_count_mismatch_alltoall(const View& v, const int count, const int comm_size, const char* fn) {
  if constexpr (metadata_checks) {
    const auto expected = static_cast<std::size_t>(count) * static_cast<std::size_t>(comm_size);
    const auto got      = v.size();

    if (got != expected) {
      std::stringstream ss;
      ss << fn << ": count mismatch: View size is " << got << ", expected count * comm_size = " << count << " * "
         << comm_size << " = " << expected;
      KokkosComm::nccl::fail_if(true, ss.str().c_str());
    }
  }
}

}  // namespace checks

}  // namespace KokkosComm::Impl
