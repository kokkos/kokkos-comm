Getting Started
===============

Quick Start
-----------

A basic build:

.. code-block:: bash

    cmake -S /path/to/kokkos-comm \
      -B /path/to/build/directory \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DKokkos_DIR=/path/to/kokkos/install/lib/cmake/Kokkos

    make -C /path/to/build/directory

    ctest -V --test-dir /path/to/build/directory

A build with ``mdspan`` support through kokkos/mdspan.
You need to turn on ONE of ``KokkosComm_USE_STD_MDSPAN`` or ``KokkosComm_USE_KOKKOS_MDSPAN``.
As of March 2024, only Clang 18 has full ``std::mdspan`` support, so you will probably want ``KokkosComm_USE_KOKKOS_MDSPAN``.

.. code-block:: bash

    cmake -S /path/to/kokkos-comm \
      -B /path/to/build/directory \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -Kokkos_ROOT=/path/to/kokkos/install/ \
      -DKokkosComm_ENABLE_MDSPAN=ON \
      -DKokkosComm_USE_KOKKOS_MDSPAN=ON


Configuration Options
---------------------

* ``KokkosComm_ENABLE_MDSPAN``: (default=OFF) build with mdspan support
  * Causes ``KOKKOSCOMM_ENABLE_MDSPAN`` to be defined in source files
* ``KokkosComm_USE_STD_MDSPAN``: (default=OFF) use std::mdspan as the mdspan implementation (if KokkosComm_ENABLE_MDSPAN)
  * Causes ``KOKKOSCOMM_USE_STD_MDSPAN`` to be defined in source files
* ``KokkosComm_USE_KOKKOS_MDSPAN``: (default=OFF) retrieve and use kokkos/mdspan as the mdspan implementation (if KokkosComm_ENABLE_MDSPAN)
  * Causes ``KOKKOSCOMM_USE_KOKKOS_MDSPAN`` to be defined in source files
  * Causes ``KOKKOSCOMM_MDSPAN_IN_EXPERIMENTAL`` to be defined in source files
* ``KokkosComm_ENABLE_PERFTESTS``: (default=ON) build performance tests
* ``KokkosComm_ENABLE_TESTS``: (default=ON) build unit tests

Known Quirks
------------

At Sandia, with the VPN enabled while using MPICH, you may have to do the following:

.. code-block:: bash

    export FI_PROVIDER=tcp
