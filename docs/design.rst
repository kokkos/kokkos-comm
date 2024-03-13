Design
======

Asynchronous MPI operations and view lifetimes
----------------------------------------------

"Immediate" functions (e.g. `isend`) return a `KokkosComm::Req`, which can be `wait()`-ed to block until the input view can be reused. `Req` also manages the lifetimes of any intermediate views needed for packing the data, releasing those views when `wait()` is complete.

Non-contiguous Data
-------------------

- Packer::MpiDatatype which just constructs an MPI Datatype matching the mdspan and hands it off to MPI to deal with the non-contiguous data
- Packer::DeepCopy uses `Kokkos::deep_copy` to handle packing and unpacking of non-contiguous `Kokkos::View`. This requires an intermediate allocation, which only works for Kokkos Views, see `Device Data`_.

Device Data
-----------

Contiguous device data is handed to MPI as-is.

For non-contiguous Kokkos::Views in a non-``Kokkos::HostSpace``, any temporary buffers are allocated in the same memory space as the view being operated on.

For non-contiguous mdspan, there is no standards-compliant way to get an allocator that can produce the same kind of allocation as the mdspan.
In that case, the Packer::MpiDatatype packer needs to be used, where a datatype is created to describe the mdspan (without accessing any of the mdspan's data!) and then that is handed off to the MPI implementation.