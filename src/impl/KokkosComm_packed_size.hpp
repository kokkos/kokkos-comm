#pragma once

#include <cstdint>

namespace KokkosComm::Impl {

template <typename View> size_t packed_size(const View &v) { return v.size(); }

} // namespace KokkosComm::Impl