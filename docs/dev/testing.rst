Testing
===========================

Testing the Install
--------------------

1. Do a standard KokkosComm build and install. This build does not need to include unit tests or perf tests -- they will later be compiled against this install.
2. Do another build, this time with ``-DKokkosComm_INSTALL_TESTING=ON``, aimed at the previous install. This builds the unit tests after using ``find_package`` to find KokkosComm, rather than defining KokkosComm itself.
3. Run the tests of this second build.

.. code-block:: bash

    export KC_SRC=path/to/kokkos-comm
    export K_INSTALL=path/to/kokkos/install

    export KC_BUILD="$KC_SRC"/build
    export KC_INSTALL="$KC_SRC"/install
    export KC_INSTALL_BUILD="$KC_SRC"/install-build

    rm -rf "$KC_BUILD"
    rm -rf "$KC_INSTALL"
    rm -rf "$KC_INSTALL_BUILD"

    cmake -S "$KC_SRC" -B "$KC_BUILD" \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DKokkos_ROOT=$K_INSTALL \
      -DKokkosComm_ENABLE_TESTS=OFF \
      -DKokkosComm_ENABLE_PERFTESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$KC_INSTALL

    cmake  --build $KC_BUILD --target install --parallel 16

    cmake -S "$KC_SRC" -B "$KC_INSTALL_BUILD" \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DKokkos_ROOT=$K_INSTALL \
      -DCMAKE_PREFIX_PATH=$KC_INSTALL \
      -DKokkosComm_INSTALL_TESTING=ON

    VERBOSE=1 cmake --build "$KC_INSTALL_BUILD"

    ctest -V --test-dir "$KC_INSTALL_BUILD ""   


    