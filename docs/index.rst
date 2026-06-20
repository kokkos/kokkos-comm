***********
Kokkos Comm
***********

Kokkos Comm provides experimental explicit communication interfaces for distributed applications using the Kokkos C++ Performance Portability Programming ecosystem.

.. warning:: This is a work in progress and is not yet ready for production use.


Questions?
==========

Reach us on the `Kokkos Slack <https://kokkosteam.slack.com>`_ (``mpi-interop`` channel), open a discussion or file an issue on the `GitHub repository <https://github.com/kokkos/kokkos-comm/issues>`_.


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

   api/concepts
   api/traits
   api/packing
   api/core
   api/mpi
   api/nccl

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   CONTRIBUTING
   dev/impl_comm_space.rst
   dev/testing
   dev/mpi
   dev/docs


Index
=====

* :ref:`genindex`
