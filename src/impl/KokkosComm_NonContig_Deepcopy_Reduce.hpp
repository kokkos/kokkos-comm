#pragma once

#include "impl/KokkosComm_concepts.hpp"
#include "impl/KokkosComm_NonContigCtx.hpp"
#include "impl/KokkosComm_types.hpp"
#include "impl/KokkosComm_request.hpp"

// use deep-copy to make non-contiguous views a single contiguous block

namespace KokkosComm::Impl {

template <KokkosExecutionSpace Space, KokkosView SendView, KokkosView RecvView>
struct NonContigDeepCopyReduce {

  static CtxReduce pre_reduce(const Space &space, const SendView &sv, const RecvView &rv) {
    throw std::runtime_error("unimplemented");
  }

  static CtxReduce post_reduce(const Space &space, const SendView &sv, const RecvView &rv, CtxReduce &ctx) {
    throw std::runtime_error("unimplemented");
  }

};

};  // namespace KokkosComm::Impl
