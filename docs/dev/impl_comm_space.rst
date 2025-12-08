*************************************
Implementing a ``CommunicationSpace``
*************************************

`PR #109 <https://github.com/kokkos/kokkos-comm/pulls/109>`_ introduced the concept of multiple communication spaces. A ``CommunicationSpace`` defines what KokkosComm actually does when you call, e.g., ``KokkosComm::send``.

Your implementation is a struct or class that represents your communication space. Then, you need partial specializations for some associated types (handles, requests, etc.), for each core API functions.


Struct representing your ``CommunicationSpace``
===============================================

The first piece is a struct that represents your ``CommunicationSpace``. We are still a bit fuzzy on the interface for this, and ultimately, it might be very small. Its main purpose is to serve as a tag for the partial specializations of each API struct.

For example, for the MPI communication space, we define the following:

.. code-block:: cpp

    #include "KokkosComm/concepts.hpp"

    namespace KokkosComm {

    struct MpiSpace {
      ...
    };

    template <>
    struct Impl::is_communication_space<KokkosComm::Mpi> : public std::true_type {};

    } // end KokkosComm


To let core API functions know that your communication space is something KokkosComm can use to dispatch messages, you also need to declare the ``Impl::is_communication_space`` specialization using the ``CommunicationSpace`` concept.


Partial specialization of ``Handle``
=====================================

.. attention:: Section in construction...


For example, for the MPI communication space handle, we define the following:

.. code-block:: cpp

    #include "KokkosComm/concepts.hpp"
    #include "KokkosComm/mpi/comm_space.hpp"

    namespace KokkosComm {

    template <KokkosExecutionSpace ExecSpace>
    class Handle<ExecSpace, Mpi> { /* ... */ };

    } // end KokkosComm


Partial specialization of ``Req``
===================================

.. attention:: Section in construction...


For example, for the MPI communication space request, we define the following:

.. code-block:: cpp

    #include "KokkosComm/mpi/comm_space.hpp"

    namespace KokkosComm {

    template <>
    class Req<MpiSpace> { /* ... */ };

    } // end KokkosComm


Partial specialization for each API struct
===========================================

The core API functions are actually implemented by partial specializations of structs. Conceptually, there is an internal interface that needs to be satisfied for each API:

.. code-block:: cpp

    #include "KokkosComm/concepts.hpp"

    namespace KokkosComm::Impl {

    template <KokkosView RecvView, KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
    struct Recv<RecvView, ExecSpace, CommSpace> { /* ... */ };

    } // end KokkosComm::Impl


In the above, ``CommSpace`` is a type that represents the communication space implementation.
For example, for the MPI communication space, we create a partial specialization of that struct template (notice fewer template parameters and the use of the ``Mpi`` "tag" struct):

.. code-block:: cpp

    #include "KokkosComm/concepts.hpp"

    namespace KokkosComm::Impl {

    template <KokkosView RecvView, KokkosExecutionSpace ExecSpace>
    struct Recv<RecvView, ExecSpace, MpiSpace> { /* ... */ };

    } // end KokkosComm::Impl

Minimal requirements of a new communication backend
---------------------------------------------------

For now, you need to implement the following three structs to get a new backend.

.. note:: As KokkosComm develops, you may need to provide more core API structs for your communication space to qualify as a new backend.

``Send`` concept
^^^^^^^^^^^^^^^^

An asynchronous/non-blocking message send:

.. code-block:: cpp

    #include "KokkosComm/concepts.hpp"
    #include "my_comm_space.hpp"

    namespace KokkosComm::Impl {

    template <KokkosView SendView, KokkosExecutionSpace ExecSpace>
    struct Send<SendView, ExecSpace, MyCommSpace> {
      static auto execute(Handle<ExecSpace, MyCommSpace> &h, const SendView &sv, int dest) -> Req<MyCommSpace> {
        // actual implementation of `send` with your communication backend
      }
    };

    } // end KokkosComm::Impl


``Recv`` concept
^^^^^^^^^^^^^^^^

An asynchronous/non-blocking message receive.

.. code-block:: cpp

    #include "KokkosComm/concepts.hpp"
    #include "my_comm_space.hpp"

    namespace KokkosComm::Impl {

    template <KokkosView RecvView, KokkosExecutionSpace ExecSpace>
    struct Recv<RecvView, ExecSpace, MyCommSpace> {
      static auto execute(Handle<ExecSpace, MyCommSpace> &h, const RecvView &sv, int src) -> Req<MyCommSpace> {
        // actual implementation of `recv` with your communication backend
      }
    };

    } // end KokkosComm::Impl


``Barrier`` concept
^^^^^^^^^^^^^^^^^^^

A global barrier.

.. code-block:: cpp

    #include "KokkosComm/concepts.hpp"
    #include "my_comm_space.hpp"

    namespace KokkosComm::Impl {

    template <KokkosExecutionSpace ExecSpace>
    struct Recv<ExecSpace, MyCommSpace> {
      static auto execute(Handle<ExecSpace, MyCommSpace> &&h) -> Req<MyCommSpace> {
        // actual implementation of `barrier` with your communication backend
      }
    };

    } // end KokkosComm::Impl
