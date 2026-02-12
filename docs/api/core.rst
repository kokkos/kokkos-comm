****
Core
****

Point-to-point
==============

.. cpp:namespace:: KokkosComm

Send
----

.. warning:: This is not a blocking operation despite being named like ``MPI_Send``.

.. cpp:function:: template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto send(Handle<ExecSpace, CommSpace> &h, SendView &sv, int dest) -> Req<CommSpace>

    Initiates a non-blocking send operation.

    :tparam SendView: The type of the Kokkos view to send.
    :tparam ExecSpace: The execution space to use. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param h: A handle to the execution space and transport mechanism.
    :param sv: The Kokkos view to send.
    :param dest: The destination rank.

    :return: A request object of type ``Req<CommSpace>`` representing the non-blocking send operation.


.. cpp:function:: template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto send(SendView &sv, int dest) -> Req<CommSpace>

    Initiates a non-blocking send operation using a default handle.

    :tparam SendView: The type of the Kokkos view to send.
    :tparam ExecSpace: The execution space to use. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param sv: The Kokkos view to send.
    :param dest: The destination rank.

    :return: A request object of type ``Req<CommSpace>`` representing the non-blocking send operation.

**Example usage:**

.. literalinclude:: core_send.cpp
    :language: cpp

Receive
-------

.. warning:: This is not a blocking operation despite being named like ``MPI_Recv``.

.. cpp:function:: template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto recv(Handle<ExecSpace, CommSpace> &h, RecvView &sv, int dest) -> Req<CommSpace>

    Initiates a non-blocking receive operation.

    :tparam RecvView: The type of the Kokkos view for receiving data.
    :tparam ExecSpace: The execution space where the operation will be performed. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param h: A handle to the execution space and transport mechanism.
    :param rv: The Kokkos view where the received data will be stored.
    :param src: The source rank from which to receive data.

    :return: A request object of type ``Req<CommSpace>`` representing the non-blocking receive operation.

    This function initiates a non-blocking receive operation using the specified execution space and transport mechanism. The data will be received into the provided view from the specified source rank and message tag. The function returns a request object that can be used to check the status of the receive operation or to wait for its completion.


.. cpp:function:: template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto recv(RecvView &sv, int dest) -> Req<CommSpace>

    Initiates a non-blocking receive operation using a default handle.

    :tparam RecvView: The type of the Kokkos view for receiving data.
    :tparam ExecSpace: The execution space where the operation will be performed. Defaults to `Kokkos::DefaultExecutionSpace`.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param rv: The Kokkos view where the received data will be stored.
    :param src: The source rank from which to receive data.

    :return: A request object of type ``Req<CommSpace>`` representing the non-blocking receive operation.

**Example usage:**

.. literalinclude:: core_recv.cpp
   :language: cpp


Collectives
===========

.. important:: 
    Collective operations act **element-wise** on the input Views. Multi-dimensional Views are treated as a **logically flattened** sequence of values, and the reduction is applied over that sequence. All participating Views must have **identical extents**; mismatched shapes result in undefined behavior.

    The reduction operator must be **associative**, but ordering of partial combinations is **not guaranteed**, and the operation is not required to be commutative.

.. cpp:namespace:: KokkosComm

.. cpp:function:: template <KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto barrier(Handle<ExecSpace, CommSpace> &&h) -> void

    A function to create a barrier using the given execution space and transport handle.

    :tparam ExecSpace: The execution space where the operation will be performed. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param h: A handle of type ``Handle<ExecSpace, CommSpace>`` to be forwarded to the barrier implementation.


Related types
=============

.. cpp:namespace:: KokkosComm

.. cpp:class:: template <CommunicationSpace CommSpace = DefaultCommSpace> Req

    A template class to handle requests with different communication backend types.

    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.
