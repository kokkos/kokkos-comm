# kokkos-comm
A toy MPI wrapper for [Kokkos](https://github.com/kokkos/kokkos).

## Getting Started

```
mkdir -p build && cd build
cmake ..
make
ctest
```

## Design

- [x] use `Kokkos::deep_copy` to handle packing and unpacking of non-contiguous views
  - When non-contiguous views are passed to an MPI function, a temporary contiguous view of matching extent is allocated, and `Kokkos::deep_copy` is used to pack the data.
- [x] "Immediate" functions (e.g. `isend`) return a `KokkosComm::Req`, which can be `wait()`-ed to block until the input view can be reused. `Req` also manages the lifetimes of any intermediate views needed for packing the data, releasing those views when `wait()` is complete.


## Considerations

- MPI threaded-ness and Kokkos backends (Serial with multiple instances, Threads, etc)
- Are there circumstances in which we can fuse packing into another kernel?
- A better pack/unpack interface
  - implement these side-by-side since they're duals
  - allow something like "pack this type into half_t, other types into the same scalar"
- More convenient collective wrappers
  - Outer dimension has destination rank?