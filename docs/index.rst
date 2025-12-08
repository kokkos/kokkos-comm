*******************************************************************
KokkosComm: Communication layer for distributed Kokkos applications
*******************************************************************

KokkosComm provides experimental MPI interfaces (and more!) for the Kokkos C++ Performance Portability Programming ecosystem.

.. warning:: This is a work in progress and is not yet ready for general use.


Questions?
==========

Find us on `Slack <https://kokkosteam.slack.com>`_ (``mpi-interop`` channel) or open an issue on `GitHub <https://github.com/kokkos/kokkos-comm/issues>`_.


Documentation Content
=====================

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting_started/setup
   getting_started/perf_tests

.. toctree::
   :maxdepth: 1
   :caption: Design Model

   design/overview
   design/mpi_interop
   design/nccl_interop

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/core
   api/traits
   api/packing
   api/mpi

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   CONTRIBUTING
   dev/impl_comm_space.rst
   dev/testing
   dev/mpi
   dev/docs


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
