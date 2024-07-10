Core
====

Point-to-point
--------------

.. cpp:namespace:: KokkosComm

.. cpp:function:: template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, Transport TRANSPORT = DefaultTransport> Req<TRANSPORT> isend(Handle<ExecSpace, TRANSPORT> &h, SendView &sv, int dest, int tag)

  Initiates a non-blocking send operation.

  :tparam SendView: The type of the Kokkos view to send.
  :tparam ExecSpace: The execution space to use. Defaults to Kokkos::DefaultExecutionSpace.
  :tparam TRANSPORT: The transport mechanism to use. Defaults to DefaultTransport.

  :param h: A handle to the execution space and transport mechanism.
  :param sv: The Kokkos view to send.
  :param dest: The destination rank.
  :param tag: The message tag.

  :return: A request object for the non-blocking send operation.

.. cpp:function:: template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, Transport TRANSPORT = DefaultTransport> Req<TRANSPORT> isend(SendView &sv, int dest, int tag)

   Initiates a non-blocking send operation using a default handle.

   :tparam SendView: The type of the Kokkos view to send.
   :tparam ExecSpace: The execution space to use. Defaults to Kokkos::DefaultExecutionSpace.
   :tparam TRANSPORT: The transport mechanism to use. Defaults to DefaultTransport.

   :param sv: The Kokkos view to send.
   :param dest: The destination rank.
   :param tag: The message tag.

   :return: A request object for the non-blocking send operation.

   Example usage:

.. literalinclude:: core_isend.cpp
   :language: cpp



.. cpp:function:: template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, Transport TRANSPORT = DefaultTransport> Req<TRANSPORT> irecv(Handle<ExecSpace, TRANSPORT> &h, RecvView &rv, int src, int tag)

   Initiates a non-blocking receive operation.

   :tparam RecvView: The type of the Kokkos view for receiving data.
   :tparam ExecSpace: The execution space where the operation will be performed. Defaults to `Kokkos::DefaultExecutionSpace`.
   :tparam TRANSPORT: The transport mechanism to be used. Defaults to `DefaultTransport`.

   :param h: A handle to the execution space and transport mechanism.
   :param rv: The Kokkos view where the received data will be stored.
   :param src: The source rank from which to receive data.
   :param tag: The message tag to identify the communication.

   :return: A request object of type `Req<TRANSPORT>` representing the non-blocking receive operation.

   This function initiates a non-blocking receive operation using the specified execution space and transport mechanism. The data will be received into the provided view from the specified source rank and message tag. The function returns a request object that can be used to check the status of the receive operation or to wait for its completion.

   Example usage:

.. literalinclude:: core_irecv.cpp
   :language: cpp





.. cpp:function:: template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, Transport TRANSPORT = DefaultTransport> Req<TRANSPORT> irecv(RecvView &rv, int src, int tag)

   Initiates a non-blocking receive operation using a default handle.

   :tparam RecvView: The type of the Kokkos view for receiving data.
   :tparam ExecSpace: The execution space where the operation will be performed. Defaults to `Kokkos::DefaultExecutionSpace`.
   :tparam TRANSPORT: The transport mechanism to be used. Defaults to `DefaultTransport`.

   :param rv: The Kokkos view where the received data will be stored.
   :param src: The source rank from which to receive data.
   :param tag: The message tag to identify the communication.

   :return: A request object of type `Req<TRANSPORT>` representing the non-blocking receive operation.


Collective
----------

.. cpp:namespace:: KokkosComm

.. cpp:function:: template <KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, Transport TRANSPORT = DefaultTransport> void barrier(Handle<ExecSpace, TRANSPORT> &&h)

   A function to create a barrier using the given execution space and transport handle.

   :tparam ExecSpace: The execution space to be used. Defaults to `Kokkos::DefaultExecutionSpace`.
   :tparam TRANSPORT: The transport mechanism to be used. Defaults to `DefaultTransport`.
   :param h: A handle of type `Handle<ExecSpace, TRANSPORT>` to be forwarded to the barrier implementation.



Related Types
-------------

.. cpp:namespace:: KokkosComm

.. cpp:class:: template <Transport TRANSPORT = DefaultTransport> Req

   A template class to handle requests with different transport types.

   :tparam TRANSPORT: The type of transport. Defaults to :cpp:enumerator:`KokkosComm::DefaultTransport`.
