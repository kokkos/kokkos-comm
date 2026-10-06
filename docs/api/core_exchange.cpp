// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

auto exec = Kokkos::DefaultExecutionSpace{};
auto comm = KokkosComm::Communicator<>::from_raw(raw_comm_handle, exec);

// Interior columns are 1 through ncols; columns 0 and ncols + 1 are halos.
const int nrows = 128;
const int ncols = 64;
Kokkos::View<double**, Kokkos::LayoutRight> field("field", nrows, ncols + 2);
Kokkos::deep_copy(exec, field, static_cast<double>(comm.rank()));

const int dst = (comm.rank() + 1) % comm.size();
const int src = (comm.rank() + comm.size() - 1) % comm.size();

// Columns of a LayoutRight view are non-contiguous.
auto right_face = Kokkos::subview(field, Kokkos::ALL, ncols);
auto left_halo  = Kokkos::subview(field, Kokkos::ALL, 0);
auto req = KokkosComm::exchange(comm, right_face, dst, left_halo, src);

// Independent interior work can be done here without touching these columns.
req.wait();
// Both transfers and receive unpacking are complete; left_halo is ready to use.
