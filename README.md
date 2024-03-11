# kokkos-mpi

> [!WARNING]
> UNOFFICIAL MPI interfaces for [Kokkos](https://github.com/kokkos/kokkos) C++ Performance Portability Programming EcoSystem

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

[cwpearson.github.io/kokkos-mpi/](https://cwpearson.github.io/kokkos-mpi/)

https://www.sphinx-doc.org/en/master/usage/domains/cpp.html


## Design

| | `Kokkos::View` | `mdspan` |
|-|-|-|
| MPI_Isend  | x | x |
| MPI_Recv   | x | x |
| MPI_Send   | x |   |
| MPI_Reduce |   |   |

- [ ] Grab and reuse Kokkos Core configuration
- [ ] MPI Communicator wrapper

- [ ] Packing
  - [ ] first pass could be a MpiDatatypePacker which just constructs an MPI Datatype matching the mdspan and hands it off to MPI to deal with the non-contiguous data
  - [ ] second pass would be to somehow associate a Kokkos memory space with the `mdspan` so we know how to allocate intermediate packing buffers
- [x] use `Kokkos::deep_copy` to handle packing and unpacking of non-contiguous `Kokkos::View`
  - When non-contiguous views are passed to an MPI function, a temporary contiguous view of matching extent is allocated, and `Kokkos::deep_copy` is used to pack the data.
- [x] "Immediate" functions (e.g. `isend`) return a `KokkosComm::Req`, which can be `wait()`-ed to block until the input view can be reused. `Req` also manages the lifetimes of any intermediate views needed for packing the data, releasing those views when `wait()` is complete.
- [x] `KokkosComm::Traits` is specialized for `Kokkos::View` and `mdspan`
  - whether `View` needs to be packed or not
  - what `pack` does for `View`
  - what `unpack` does for `View`
  - spans (distance between beginning of first byte and end of last byte)
- [ ] Future work
  - [x] host data `mdspan` 
  - [ ] device data `mdspan`

## Considerations

- macOS xcode 15.3 doesn't support `std::mdspan`, so we use `kokkos/mdspan`.
- Pluggable packing strategies
  - This would probably be a template parameter on the interface, which would be specialized to actually implement the various MPI operations
  - Constructing matching MPI datatype and sending
  - Packing into a contiguous buffer and sending
- How to handle discriminate between mdspans of host or device data, for packing
  - Alternatively, construct an MPI datatype matching the non-contiguous sp
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
clang-format-8 -i {src,unit_tests,perf_tests}/**/*.[ch]pp
```
