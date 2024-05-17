#pragma once

#include "impl/KokkosComm_NonContig_Deepcopy_SendRecv.hpp"
#include "impl/KokkosComm_NonContig_Deepcopy_Reduce.hpp"
#include "impl/KokkosComm_NonContig_Deepcopy_Alltoall.hpp"
// #include "impl/KokkosComm_NonContig_MultiMessage.hpp"
// #include "impl/KokkosComm_NonContig_Datatypes.hpp"

namespace KokkosComm {

#if 1
template <KokkosExecutionSpace Space, KokkosView View>
using DefaultNonContigSendRecv = KokkosComm::Impl::NonContigDeepCopySendRecv<Space, View>;

template <KokkosExecutionSpace Space, KokkosView SendView, KokkosView RecvView>
using DefaultNonContigReduce = KokkosComm::Impl::NonContigDeepCopyReduce<Space, SendView, RecvView>;

template <KokkosExecutionSpace Space, KokkosView SendView, KokkosView RecvView>
using DefaultNonContigAlltoall = KokkosComm::Impl::NonContigDeepCopyAlltoall<Space, SendView, RecvView>;
#elif 0
template <KokkosExecutionSpace Space, KokkosView View>
using DefaultNonContig = KokkosComm::Impl::NonContigMultiMessage<Space, View>;
#else
template <KokkosExecutionSpace Space, KokkosView View>
using DefaultNonContig = KokkosComm::Impl::NonContigDatatypes<Space, View>;
#endif

}  // namespace KokkosComm