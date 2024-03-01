# kokkos-comm
A toy MPI wrapper for [Kokkos](https://github.com/kokkos/kokkos).

## Getting Started

* **Requires c++20**
* **Requires c++23 for std::mdspan**

### macOS

macOS standard toolchain has incomplete `std::mdspan` support: `std::layout_stride` is not implemented as of xcode 15.3 release candidates

At sandia, with MPICH and the VPN enabled, you may need to do this before running any tests:
```bash
export FI_PROVIDER=tcp
```


```
mkdir -p build && cd build
cmake .. \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DKokkosComm_ENABLE_MDSPAN=ON \
  -DKokkosComm_USE_KOKKOS_MDSPAN=ON \
  -DKokkos_DIR=/path/to/kokkos-install/lib/cmake/Kokkos
make
ctest
```

## Documentation

[cwpearson.github.io/kokkos-comm/](https://cwpearson.github.io/kokkos-comm/)

https://www.sphinx-doc.org/en/master/usage/domains/cpp.html



## Design
- [ ] Overloads for `Kokkos::view`
- [ ] Overloads of `std::mdspan`
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

## Contributing

```bash
shopt -s globstar
clang-format-8 -i {src,unit_tests,perf_tests}/**/*.{c,h}pp
```