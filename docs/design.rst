Design
======

Asynchronous MPI operations, View Lifetimes, and Packing
--------------------------------------------------------

Asynchronous MPI operations use an ``MPI_Request``, which is a handle that can be used to refer to the operation later (e.g., ``MPI_Wait``).

KokkosComm has an analogous concept, `KokkosComm::Req`.
"Immediate" functions (e.g. `isend`) return a `KokkosComm::Req`, which can be `wait()`-ed to block until the input view can be reused.

There are three consequences

First, to ensure compatibility with MPI semantics, KokkosComm immediate functions will call the corresponding MPI function before they return.

Second, the KokkosComm packing strategy may require that an intermediate view be allocated, and this view needs to have a lifetime at least as long as the communication.
The ``KokkosComm::Req::keep_until_wait`` interface allows the `KokkosComm::Req` to hold those views until ``wait`` is called.

Third, for asynchronous receive operations, the packing strategy may require that the buffer provided by the underlying MPI operation be further unpacked.
The ``KokkosComm::Req::call_and_drop_at_wait`` allows the `KokkosComm::Req` to execute (and then drop) callback functors when ``wait`` is called.
For example, `KokkosComm::irecv` uses this functionality to attach an unpacking operation to the `KokkosComm::Req::wait` call.

Non-contiguous Data
-------------------

- Packer::DeepCopy uses `Kokkos::deep_copy` to handle packing and unpacking of non-contiguous `Kokkos::View`. This requires an intermediate allocation, which only works for Kokkos Views, see `Device Data`_.

Device Data
-----------

Contiguous device data is handed to MPI as-is.

For non-contiguous Kokkos::Views in a non-``Kokkos::HostSpace``, any temporary buffers are allocated in the same memory space as the view being operated on.
