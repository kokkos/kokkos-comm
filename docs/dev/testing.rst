Testing
===========================

The unit tests (``unit_tests``) and perf tests (``perf_tests``) are conceptually maintained as separate CMake projects contained in the same source tree.
This enforces some discipline with respect to testing the KokkosComm installation, since the tests don't have any special privileges.
During a build with tests enabled, they just get included at the very end of the build.


Testing the Install
--------------------

1. Do a standard KokkosComm build and install with tests disabled. This build does not need to include unit tests or perf tests -- they will later be compiled against this install.
2. Do another build with the ``unit_tests`` directory as the source. You will need to provide ``-DKokkosComm_ROOT=/path/to/KokkosCommInstall`` just as any other external project would
3. Run the tests on that build
4. Repeat step 2 but with ``perf_tests`` as the source directory
5. Run the tests on that build

.. code-block:: bash

    set -eou pipefail

    export KOKKOS_SRC="$PWD"/kokkos
    export KOKKOS_BUILD="$PWD"/kokkos-build
    export KOKKOS_INSTALL="$PWD"/kokkos-install
    export COMM_SRC="$PWD"
    export COMM_BUILD=build
    export COMM_INSTALL="$PWD"/install
    export COMM_UNIT_TESTS_BUILD=unit-tests-build
    export COMM_PERF_TESTS_BUILD=perf-tests-build

    echo "==== CLONE KOKKOS ===="
    git clone https://github.com/kokkos/kokkos.git --branch master --depth 1 "$KOKKOS_SRC" || true

    echo "==== CONFIGURE KOKKOS ===="
    cmake -S "$KOKKOS_SRC" -B "$KOKKOS_BUILD" -DCMAKE_INSTALL_PREFIX="$KOKKOS_INSTALL" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON

    echo "==== BUILD & INSTALL KOKKOS ===="
    cmake --build "$KOKKOS_BUILD" --parallel $(nproc) -t install

    echo "==== CFG KOKKOS COMM ===="
    rm -rf "$COMM_BUILD"
    cmake -S "$COMM_SRC" -B "$COMM_BUILD" -DKokkos_ROOT="$KOKKOS_INSTALL" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DKokkosComm_ENABLE_TESTS=OFF -DKokkosComm_ENABLE_PERFTESTS=OFF -DCMAKE_INSTALL_PREFIX="$COMM_INSTALL"

    echo "==== BUILD & INSTALL KOKKOS COMM ===="
    rm -rf "$COMM_INSTALL"
    VERBOSE=1 cmake --build "$COMM_BUILD" --target install

    echo "==== REMOVE KOKKOS COMM BUILD FILES ===="
    rm -rf $COMM_BUILD

    echo "==== CFG UNIT TESTS ===="
    rm -rf "$COMM_UNIT_TESTS_BUILD"
    cmake -S "$COMM_SRC"/unit_tests -B "$COMM_UNIT_TESTS_BUILD" -DKokkos_ROOT="$KOKKOS_INSTALL" -DKokkosComm_ROOT="$COMM_INSTALL" -DCMAKE_BUILD_TYPE=RelWithDebInfo

    echo "==== BUILD UNIT TESTS ===="
    VERBOSE=1 cmake --build "$COMM_UNIT_TESTS_BUILD" --parallel $(nproc)

    echo "==== RUN UNIT TESTS ===="
    ctest -V --test-dir "$COMM_UNIT_TESTS_BUILD"

    echo "==== CFG PERF TESTS ===="
    rm -rf "$COMM_PERF_TESTS_BUILD"
    cmake -S "$COMM_SRC"/perf_tests -B "$COMM_PERF_TESTS_BUILD" -DKokkos_ROOT="$KOKKOS_INSTALL" -DKokkosComm_ROOT="$COMM_INSTALL" -DCMAKE_BUILD_TYPE=RelWithDebInfo

    echo "==== BUILD PERF TESTS ===="
    VERBOSE=1 cmake --build "$COMM_PERF_TESTS_BUILD"  --parallel $(nproc)

    echo "==== RUN PERF TESTS ===="
    ctest -V --test-dir "$COMM_PERF_TESTS_BUILD"
