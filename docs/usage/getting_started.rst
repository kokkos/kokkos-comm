Getting Started
===============

Quick Start
-----------

A basic build:

.. code-block:: bash

    cmake -S /path/to/kokkos-comm \
      -B /path/to/build/directory \
      -DKokkos_ROOT=/path/to/kokkos-install

    make -C /path/to/build/directory

    ctest -V --test-dir /path/to/build/directory

Configuration Options
---------------------

* ``KokkosComm_ENABLE_PERFTESTS``: (default=ON) build performance tests
* ``KokkosComm_ENABLE_TESTS``: (default=ON) build unit tests

Known Quirks
------------

At Sandia, with the VPN enabled while using MPICH, you may have to do the following:

.. code-block:: bash

    export FI_PROVIDER=tcp
