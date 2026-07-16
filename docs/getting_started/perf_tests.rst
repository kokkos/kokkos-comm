**********
Benchmarks
**********

Benchmarks are implemented in the ``perf_tests`` directory. Currently, all benchmarks use the MPI backend.

Available benchmarks:

.. list-table::
    :header-rows: 1
    :align: left
    :widths: 40 80

    * - File name
      - Description

    * - ``test_2d_halo.cpp``
      - A 2D halo exchange

    * - ``test_sendrecv.cpp``
      - A ping-pong between ranks 0 and 1

    * - ``test_channel_sendrecv.cpp``
      - A send/receive between neighboring ranks using ``KokkosComm::Channel``

    * - ``test_osu_latency.cpp``
      - An implementation of the OSU latency benchmark (comparison between Kokkos Comm and raw MPI)


To build and run the benchmarks, see the `dedicated page on testing <../dev/testing.html>`_:
