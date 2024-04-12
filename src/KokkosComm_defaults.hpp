#pragma once

#include "impl/KokkosComm_NonContig_Deepcopy.hpp"
#include "impl/KokkosComm_NonContig_MultiMessage.hpp"
#include "impl/KokkosComm_NonContig_Datatypes.hpp"

namespace KokkosComm {

#if 0
template <KokkosExecutionSpace Space, KokkosView View>
using DefaultNonContig = KokkosComm::Impl::NonContigDeepCopy<Space, View>;
#elif 0
template <KokkosExecutionSpace Space, KokkosView View>
using DefaultNonContig = KokkosComm::Impl::NonContigMultiMessage<Space, View>;
#else
template <KokkosExecutionSpace Space, KokkosView View>
using DefaultNonContig = KokkosComm::Impl::NonContigDatatypes<Space, View>;
#endif

}  // namespace KokkosComm