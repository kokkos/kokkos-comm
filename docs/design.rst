Design
======

Asynchronous MPI operations and view lifetimes
----------------------------------------------

"Immediate" functions (e.g. `isend`) return a `KokkosComm::Req`, which can be `wait()`-ed to block until the input view can be reused. `Req` also manages the lifetimes of any intermediate views needed for packing the data, releasing those views when `wait()` is complete.

Non-contiguous Data
-------------------

- Packer::DeepCopy uses `Kokkos::deep_copy` to handle packing and unpacking of non-contiguous `Kokkos::View`. This requires an intermediate allocation, which only works for Kokkos Views, see `Device Data`_.

Device Data
-----------

Contiguous device data is handed to MPI as-is.

For non-contiguous Kokkos::Views in a non-``Kokkos::HostSpace``, any temporary buffers are allocated in the same memory space as the view being operated on.
