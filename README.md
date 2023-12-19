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
- [x] `KokkosComm::Traits<View>` can be specialized for a `View`:
  - whether `View` needs to be packed or not
  - what `pack` does for `View`
  - what `unpack` does for `View`

## Considerations

- MPI threaded-ness and Kokkos backends (Serial with multiple instances, Threads, etc)
- Are there circumstances in which we can fuse packing into another kernel?
- A better pack/unpack interface
  - Maybe a `PackTraits<View>` where users can specialize `PackTraits` for any types they want to handle
  - Also, could introduce a runtime packing argument to the various functions, like a pack tag
- More convenient collective wrappers
  - Outer dimension has destination rank?
- Custom reductions?

## Performance Tests

* `test_2dhalo.cpp`: a 2d halo exchange
* `test_sendrecv.cpp`: ping-pong between ranks 0 and 1