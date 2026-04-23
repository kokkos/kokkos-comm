#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

// Create an execution space instance
auto exec = Kokkos::DefaultExecutionSpace();
// Create a communicator
auto comm = KokkosComm::Communicator<>::duplicate_from_raw(raw_comm_handle, exec).value();

// Allocate a view to receive the data
Kokkos::View<double*> data("recv_view", 100);

// Source rank
int src_rank = 1;

// Initiate a non-blocking receive with a handle
auto req1 = KokkosComm::recv(comm, data, src_rank);

// Simulate a blocking receive by waiting immediately
KokkosComm::recv(comm, data, src_rank).wait();

// Wait for a requests to complete
KokkosComm::wait(req1);
