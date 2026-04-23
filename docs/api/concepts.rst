********
Concepts
********

.. cpp:namespace:: KokkosComm

Kokkos Comm-specific concepts
=============================

.. cpp:concept:: template <typename T> CommunicationSpace

    Specifies that a type ``T`` is a Kokkos Comm communication backend.

    **Constraints for implementing the ``CommunicationSpace`` concept**

    .. cpp:type:: T::communication_space

    .. cpp:type:: T::communicator_type

    .. cpp:type:: T::request_type

    .. cpp:type:: T::datatype_type

    .. cpp:type:: T::reduction_op_type

    .. cpp:type:: T::size_type

    .. cpp:type:: T::rank_type

Types implementing the ``CommunicationSpace`` concept
"""""""""""""""""""""""""""""""""""""""""""""""""""""

.. cpp:class:: MpiSpace

    .. cpp:type:: communication_space = MpiSpace

    .. cpp:type:: communicator_type = MPI_Comm

    .. cpp:type:: request_type = MPI_Request

    .. cpp:type:: datatype_type = MPI_Datatype

    .. cpp:type:: reduction_op_type = MPI_Op

    .. cpp:type:: size_type = int

    .. cpp:type:: rank_type = int

.. cpp:class:: Experimental::NcclSpace

    .. cpp:type:: communication_space = NcclSpace

    .. cpp:type:: communicator_type = ncclComm_t

    .. cpp:type:: request_type = cudaEvent_t

    .. cpp:type:: datatype_type = ncclDataType_t

    .. cpp:type:: reduction_op_type = ncclRedOp_t

    .. cpp:type:: size_type = int

    .. cpp:type:: rank_type = int


.. cpp:concept:: template <typename T> ReductionOperator

    Specifies that a type ``T`` is a Kokkos Comm reduction operator.

Types implementing the ``ReductionOperator`` concept
""""""""""""""""""""""""""""""""""""""""""""""""""""

.. cpp:class:: BAnd

    The bitwise AND operator (``&``).

.. cpp:class:: BOr

    The bitwise OR operator (``|``).

.. cpp:class:: BXor

    The bitwise XOR (exclusive OR) operator (``^``).

.. cpp:class:: LAnd

    The logical AND operator (``&&``).

.. cpp:class:: LOr

    The logical OR operator (``||``).

.. cpp:class:: LXor

    The logical XOR (exclusive OR) operator.

.. cpp:class:: Max

    The MAXIMUM operator.

.. cpp:class:: MaxLoc

    The MAXIMUM with location (index) operator.

.. cpp:class:: Min

    The MINIMUM operator.

.. cpp:class:: MinLoc

    The MINIMUM with location (index) operator.

.. cpp:class:: Sum

    The SUM operator.

.. cpp:class:: Prod

    The PRODUCT operator.

.. cpp:class:: Average

    The AVERAGE operator.


Kokkos interoperability concepts
================================

.. cpp:concept:: template <typename T> KokkosView

    Specifies that a type ``T`` is a ``Kokkos::View`` object.


.. cpp:concept:: template <typename T> KokkosExecutionSpace

    Specifies that a type ``T`` is a ``Kokkos::ExecutionSpace``.
