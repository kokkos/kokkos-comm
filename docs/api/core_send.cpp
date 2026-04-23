#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

// Create an execution space instance
auto exec = Kokkos::DefaultExecutionSpace();
// Create a communicator
auto comm = KokkosComm::Communicator<>::duplicate_from_raw(raw_comm_handle, exec).value();

// Create a Kokkos view
Kokkos::View<double*> data("send_data", 100);

// Fill the view with some data
Kokkos::parallel_for(
    "fill_data", Kokkos::RangePolicy(exec, 0, 100), KOKKOS_LAMBDA(int i) { data(i) = static_cast<double>(i); }
);
exec.fence();

// Destination rank
int dst_rank = 1;

// Initiate a non-blocking send with a handle
auto req1 = KokkosComm::send(comm, data, dst_rank);

// Simulate a blocking send by waiting immediately
KokkosComm::send(comm, data, dst_rank).wait();

// Wait for a request to complete
KokkosComm::wait(req1);
