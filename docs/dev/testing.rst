*******
Testing
*******

The unit tests (``unit_tests``) and performance tests/benchmarks (``perf_tests``) are conceptually maintained as separate CMake projects contained in the same source tree.
This enforces some discipline with respect to testing the Kokkos Comm installation, since the tests don't have any special privileges.
During a build with tests enabled, they just get included at the very end of the build.


Testing the install
====================

1. Do a standard Kokkos Comm build and install with tests disabled (see the `Setup section <../getting_started/setup.html>`_). This build does not need to include unit tests or performance tests -- they will later be compiled against this install.

2. Do another build with the ``unit_tests`` directory as the source. You will need to provide ``-DKokkosComm_ROOT=<KOKKOSCOMM_INSTALL_DIR>/lib64/cmake`` just as any other external project would.

3. Run the tests on that build.

4. Repeat step 2 but with ``perf_tests`` as the source directory.

5. Run the tests on that build.

Complete workflow
-----------------

.. tip::

    For faster incremental builds, it is advised to use Ninja as your CMake generator. You can do so by passing ``-G Ninja`` when configuring the project.

    You can also have multiple build types configured simultaneously by using ``-G 'Ninja Multi-Config'`` and passing ``--config <BUILD_TYPE>`` when building the project:

    .. code-block:: console

        $ cmake -B <BUILD_DIR> -G 'Ninja Multi-Config' -DKokkos_ROOT=<KOKKOS_INSTALL_DIR> -DKokkosComm_ENABLE_TESTS=ON
        $ cmake --build <BUILD_DIR> --config <BUILD_TYPE>


.. code-block:: console

    $ # Catch failing commands
    $ set -eou pipefail

    $ # Clone Kokkos Comm
    $ git clone https://github.com/kokkos/kokkos-comm.git
    $ cd kokkos-comm

    $ # Set the required path variables
    $ export KOKKOS_SRC_DIR="$PWD/kokkos"
    $ export KOKKOS_BUILD_DIR="$PWD/kokkos-build"
    $ export KOKKOS_INSTALL_DIR="$PWD/kokkos-install"
    $ export KOKKOSCOMM_SRC_DIR="$PWD"
    $ export KOKKOSCOMM_BUILD_DIR="$PWD/build"
    $ export KOKKOSCOMM_INSTALL_DIR="$PWD/install"
    $ export KOKKOSCOMM_UNIT_TESTS_BUILD_DIR="$PWD/build-unit-tests"
    $ export KOKKOSCOMM_PERF_TESTS_BUILD_DIR="$PWD/build-perf-tests"

    $ # Clone Kokkos
    $ git clone https://github.com/kokkos/kokkos.git --branch master --depth 1 "$KOKKOS_SRC_DIR"

    $ # Configure Kokkos
    $ cmake -S "$KOKKOS_SRC_DIR" \
            -B "$KOKKOS_BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DKokkos_ENABLE_SERIAL=ON \
            -DKokkos_ENABLE_OPENMP=ON

    $ # Build Kokkos
    $ cmake --build "$KOKKOS_BUILD_DIR" --parallel $(nproc)

    $ # Install Kokkos
    $ cmake --install "$KOKKOS_BUILD_DIR" --prefix "$KOKKOS_INSTALL_DIR"

    $ # Configure Kokkos Comm
    $ [ -d "$KOKKOSCOMM_BUILD_DIR" ] && rm --recursive --force "$KOKKOSCOMM_BUILD_DIR"
    $ cmake -S "$KOKKOSCOMM_SRC_DIR" \
            -B "$KOKKOSCOMM_BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DKokkos_ROOT="$KOKKOS_INSTALL_DIR" \
            -DKokkosComm_ENABLE_TESTS=OFF \
            -DKokkosComm_ENABLE_PERFTESTS=OFF

    $ # Build Kokkos Comm
    $ cmake --build "$KOKKOSCOMM_BUILD_DIR" --parallel $(nproc) --verbose

    $ # Install Kokkos Comm
    $ [ -d "$KOKKOSCOMM_INSTALL_DIR" ] && rm --recursive --force "$KOKKOSCOMM_INSTALL_DIR"
    $ cmake --install "$KOKKOSCOMM_BUILD_DIR" --prefix "$KOKKOSCOMM_INSTALL_DIR"

    $ # Remove Kokkos Comm build files
    $ rm --recursive --force "$KOKKOSCOMM_BUILD_DIR"

    $ # Configure Kokkos Comm unit tests
    $ [ -d "$KOKKOSCOMM_UNIT_TESTS_BUILD_DIR" ] && rm --recursive --force "$KOKKOSCOMM_UNIT_TESTS_BUILD_DIR"
    $ cmake -S "$KOKKOSCOMM_SRC_DIR"/unit_tests \
            -B "$KOKKOSCOMM_UNIT_TESTS_BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DKokkos_ROOT="$KOKKOS_INSTALL_DIR" \
            -DKokkosComm_ROOT="$KOKKOSCOMM_INSTALL_DIR"

    $ # Build Kokkos Comm unit tests
    $ cmake --build "$KOKKOSCOMM_UNIT_TESTS_BUILD_DIR" --parallel $(nproc) --verbose

    $ # Run Kokkos Comm unit tests
    $ ctest -V --test-dir "$KOKKOSCOMM_UNIT_TESTS_BUILD_DIR"

    $ # Configure Kokkos Comm performance tests
    $ [ -d "$KOKKOSCOMM_PERF_TESTS_BUILD_DIR" ] && rm --recursive --force "$KOKKOSCOMM_PERF_TESTS_BUILD_DIR"
    $ cmake -S "$KOKKOSCOMM_SRC_DIR"/perf_tests \
            -B "$KOKKOSCOMM_PERF_TESTS_BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DKokkos_ROOT="$KOKKOS_INSTALL_DIR" \
            -DKokkosComm_ROOT="$KOKKOSCOMM_INSTALL_DIR" \
            -DKOKKOSCOMM_ENABLE_MPI=ON # not defined if not built alongside KokkosComm

    $ # Build Kokkos Comm performance tests
    $ cmake --build "$KOKKOSCOMM_PERF_TESTS_BUILD_DIR" --parallel $(nproc) --verbose

    $ # Run Kokkos Comm performance tests
    $ ctest -V --test-dir "$KOKKOSCOMM_PERF_TESTS_BUILD_DIR"
