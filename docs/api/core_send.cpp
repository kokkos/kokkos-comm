#include "KokkosComm/KokkosComm.hpp"

// Define the execution space and transport
using ExecSpace = Kokkos::DefaultExecutionSpace;
using CommSpace = DefaultCommunicationSpace;

// Create a Kokkos view
Kokkos::View<double*> data("data", 100);

// Fill the view with some data
Kokkos::parallel_for(
    "fill_data", Kokkos::RangePolicy<ExecSpace>(0, 100), KOKKOS_LAMBDA(int i) { data(i) = static_cast<double>(i); });

// Destination rank
int dest = 1;

// Create a handle
KokkosComm::Handle<> handle;  // Same as Handle<Execspace, CommSpace>

// Initiate a non-blocking send with a handle
auto req1 = send(handle, data, dest);

// Initiate a non-blocking send with a default handle
auto req2 = send(data, dest);

// Wait for the requests to complete (assuming a wait function exists)
KokkosComm::wait(req1);
KokkosComm::wait(req2);
