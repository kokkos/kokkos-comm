# kokkos-comm
MPI Wrapper for [Kokkos](https://github.com/kokkos/kokkos).

```
mkdir -p build && cd build
cmake ..
make
ctest
```


## Considerations

- Packing / Unpacking
- MPI threaded-ness and Kokkos backends (Serial with multiple instances, Threads, etc)
- Multi-dimensional views
- Strided views
  - Use `deep_copy` into a contiguious view with matching extents
- Are there circumstances in which we can fuse packing into another kernel?
- A better pack/unpack interface
  - implement these side-by-side since they're duals
  - allow something like "pack this type into half_t, other types into the same scalar"