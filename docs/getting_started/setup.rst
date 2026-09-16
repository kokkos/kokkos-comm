*****
Setup
*****

This page is intended as a guide to jump start new Kokkos Comm users.


System requirements
===================

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Requirement

    * - CMake
      - 3.25+

    * - C++ compiler
      - Conforming to ISO C++20 standard

    * - Kokkos
      - 4.7+

    * - MPI
      - Conforming to MPI-3+ standard

    * - NCCL
      - 2.20+

    * - CUDA Toolkit (runtime)
      - Required when enabling the NCCL backend


Kokkos Comm will attempt to support the `same systems and toolchains as Kokkos <https://kokkos.org/kokkos-core-wiki/get-started/requirements.html>`_.


Installation
============

Download
--------

Kokkos Comm's source code is available on GitHub, which you can obtain using the following command:

.. code-block:: console

    $ git clone https://github.com/kokkos/kokkos-comm.git

Or using the GitHub CLI:

.. code-block:: console

    $ gh repo clone kokkos/kokkos-comm

Build
-----

A basic configure and build of Kokkos Comm:

.. code-block:: console

    $ cmake -B <BUILD_DIR> -DKokkos_ROOT=<KOKKOS_INSTALL_DIR>
    $ cmake --build <BUILD_DIR>

Test
----

.. code-block:: console

    $ cmake -B <BUILD_DIR> \
            -DKokkos_ROOT=<KOKKOS_INSTALL_DIR> \
            -DKokkosComm_ENABLE_TESTS=ON
    $ cmake --build <BUILD_DIR>
    $ ctest --test-dir <BUILD_DIR>/unit_tests

For a detailed guide on testing Kokkos Comm, please refer to the `Testing section <../dev/testing.html>`_.

Install
-------

.. code-block:: console

    $ cmake --install <BUILD_DIR> --prefix <INSTALL_DIR>


Integration in user applications
================================

Similarly to Kokkos, Kokkos Comm is packaged with a modern CMake build system.

Once installed, you can declare Kokkos Comm as a dependency of your project by adding the following line to your ``CMakeLists.txt``:

.. code-block:: cmake

    find_package(KokkosComm REQUIRED)

Then, for every executable or library in your project that depends on Kokkos Comm:

.. code-block:: cmake

    target_link_libraries(MyTarget KokkosComm::KokkosComm)


CMake configuration options
===========================

This section lists the available options to customize your Kokkos Comm build and/or installation.

.. note: CMake options are set when configuring the project and passed using the syntax ``-D<OPTION>=<VALUE>``.

.. important:: All Kokkos Comm CMake options are prefixed with ``KokkosComm_``, which is case-sensitive.

Communication backend selection
-------------------------------

You can enable communication backends by configuring with ``-DKokkosComm_ENABLE_<COMM_BACKEND>=ON``, where ``<COMM_BACKEND>`` is replaced with the specific communication backend you want to enable (e.g. ``-DKokkosComm_ENABLE_MPI=ON`` for MPI).

.. list-table::
    :widths: 40 10 70
    :header-rows: 1
    :align: left

    * - CMake option
      - Default
      - Description

    * * ``KokkosComm_ENABLE_MPI``
      * ``ON``
      * Build with MPI backend.

    * * ``KokkosComm_ENABLE_NCCL``
      * ``OFF``
      * Build with NCCL backend (experimental).

Behavior tuning
---------------

.. list-table::
    :widths: 40 10 70
    :header-rows: 1
    :align: left

    * - CMake option
      - Default
      - Description

    * * ``KokkosComm_ABORT_ON_ERROR``
      * ``OFF``
      * Runtime check failures trigger a global abort.

General options
---------------

.. list-table::
    :widths: 40 10 70
    :header-rows: 1
    :align: left

    * - CMake option
      - Default
      - Description

    * * ``KokkosComm_ENABLE_TESTS``
      * ``OFF``
      * Build unit tests.

    * * ``KokkosComm_ENABLE_PERFTESTS``
      * ``OFF``
      * Build performance tests.

    * * ``KokkosComm_INSTALL_CMAKE_PACKAGE``
      * ``${PROJECT_IS_TOP_LEVEL}``
      * Install a CMake config-file package together with the library. Defaults to ``ON`` for standalone builds, ``OFF`` otherwise. Note that library files are always installed.


Known quirks
============

At Sandia, with the VPN enabled while using MPICH, you may have to do the following:

.. code-block:: console

    $ export FI_PROVIDER=tcp
