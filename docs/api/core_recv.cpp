#include "KokkosComm/KokkosComm.hpp"

// Define the execution space and transport
using ExecSpace = Kokkos::DefaultExecutionSpace;
using CommSpace = DefaultCommunicationSpace;

// Source rank
int src = 1;

// Create a handle
KokkosComm::Handle<> handle;  // Same as Handle<Execspace, CommSpace>

// Allocate a view to receive the data
Kokkos::View<double*> data("recv_view", 100);

// Initiate a non-blocking receive with a handle
auto req1 = recv(handle, data, src);

// Initiate a non-blocking receive with a default handle
auto req2 = recv(data, src);

// Wait for the requests to complete (assuming a wait function exists)
KokkosComm::wait(req1);
KokkosComm::wait(req2);
