// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

namespace Impl {

// Workaround for `static_assert(false)` before CWG2518/P2593
template <typename T>
inline constexpr bool dependent_false = false;

}  // namespace Impl
