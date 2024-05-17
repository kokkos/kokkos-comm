#pragma once

#include "impl/KokkosComm_concepts.hpp"
#include "impl/KokkosComm_NonContigCtx.hpp"
#include "impl/KokkosComm_types.hpp"
#include "impl/KokkosComm_request.hpp"

// use deep-copy to make non-contiguous views a single contiguous block

namespace KokkosComm::Impl {

template <KokkosExecutionSpace Space, KokkosView SendView, KokkosView RecvView>
struct NonContigDeepCopyAlltoall {
  static CtxAlltoall pre_alltoall(const Space &space, const SendView &sv, const RecvView &rv) {
    throw std::runtime_error("unimplemented");
  }

  static CtxAlltoall post_alltoall(const Space &space, const RecvView &rv, CtxAlltoall &ctx) {
    throw std::runtime_error("unimplemented");
  }

  static CtxAlltoall pre_alltoall_inplace(const Space &space, const RecvView &rv) {
    throw std::runtime_error("unimplemented");
  }

  static CtxAlltoall post_alltoall_inplace(const Space &space, const RecvView &rv, CtxAlltoall &ctx) {
    throw std::runtime_error("unimplemented");
  }
};

};  // namespace KokkosComm::Impl
