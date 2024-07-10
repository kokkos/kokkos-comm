MPI
====

.. list-table:: MPI API Support
    :widths: 40 30 15
    :header-rows: 1

    * - MPI
      - ``KokkosComm::mpi::``
      - ``Kokkos::View``
    * - ``MPI_Send``
      - ``send`` or ``send<CommMode::Standard>``
      - ✓
    * - ``MPI_Rsend``
      - ``send<CommMode::Ready>``
      - ✓
    * - ``MPI_Recv``
      - ``recv``
      - ✓
    * - ``MPI_Ssend``
      - ``send<CommMode::Synchronous>``
      - ✓
    * - ``MPI_Isend``
      - ``isend`` or ``isend<CommMode::Standard>``
      - ✓
    * - ``MPI_Irsend``
      - ``isend<CommMode::Ready>``
      - ✓
    * - ``MPI_Issend``
      - ``isend<CommMode::Synchronous>``
      - ✓
    * - ``MPI_Reduce``
      - ``reduce``
      - ✓

Point-to-point
--------------

.. cpp:namespace:: KokkosComm::mpi

.. cpp:function:: template <CommMode SendMode, KokkosExecutionSpace ExecSpace, KokkosView SendView> Req<Mpi> isend(Handle<ExecSpace, Mpi> &h, const SendView &sv, int dest, int tag)

   Initiates a non-blocking send operation.

   :tparam SendMode: The communication mode.
   :tparam ExecSpace: The execution space.
   :tparam SendView: The type of the view to be sent.
   :param h: The handle for the execution space and MPI.
   :param sv: The view to be sent.
   :param dest: The destination rank.
   :param tag: The message tag.
   :return: A request object for the non-blocking send operation.

.. cpp:function:: template <KokkosView RecvView> void irecv(const RecvView &rv, int src, int tag, MPI_Comm comm, MPI_Request &req)

   Initiates a non-blocking receive operation.

   :tparam RecvView: The type of the view to be received.
   :param rv: The view to be received.
   :param src: The source rank.
   :param tag: The message tag.
   :param comm: The MPI communicator.
   :param req: The MPI request object for the non-blocking receive operation.
   :throws std::runtime_error: If the view is not contiguous.

.. cpp:function:: template <KokkosView SendView> void send(const SendView &sv, int dest, int tag, MPI_Comm comm)

   Initiates a blocking send operation.

   :tparam SendView: The type of the view to be sent.
   :param sv: The view to be sent.
   :param dest: The destination rank.
   :param tag: The message tag.
   :param comm: The MPI communicator.

.. cpp:function:: template <CommMode SendMode = CommMode::Default, KokkosExecutionSpace ExecSpace, KokkosView SendView> void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm)

   Initiates a blocking send operation with a specified execution space and communication mode.

   :tparam SendMode: The communication mode (default is CommMode::Default).
   :tparam ExecSpace: The execution space.
   :tparam SendView: The type of the view to be sent.
   :param space: The execution space.
   :param sv: The view to be sent.
   :param dest: The destination rank.
   :param tag: The message tag.
   :param comm: The MPI communicator.

.. cpp:function:: template <KokkosView RecvView> void recv(const RecvView &rv, int src, int tag, MPI_Comm comm, MPI_Status *status)

   Initiates a blocking receive operation.

   :tparam RecvView: The type of the view to be received.
   :param rv: The view to be received.
   :param src: The source rank.
   :param tag: The message tag.
   :param comm: The MPI communicator.
   :param status: The MPI status object for the blocking receive operation.

.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView RecvView> void recv(const ExecSpace &space, RecvView &rv, int src, int tag, MPI_Comm comm)

   Initiates a blocking receive operation with a specified execution space.

   :tparam ExecSpace: The execution space.
   :tparam RecvView: The type of the view to be received.
   :param space: The execution space.
   :param rv: The view to be received.
   :param src: The source rank.
   :param tag: The message tag.
   :param comm: The MPI communicator.



Collective
----------

.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView> \
                  void KokkosComm::reduce(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Op op, int root, MPI_Comm comm)

    MPI_Reduce wrapper

    :param space: The execution space to operate in
    :param sv: The data to send
    :param rv: The view to receive into
    :param op: The MPI_Op to use in the reduction
    :param root: The root rank for the reduction
    :param comm: the MPI communicator
    :tparam SendView: A Kokkos::View to send
    :tparam RecvView: A Kokkos::View to recv
    :tparam ExecSpace: A Kokkos execution space to operate in

.. cpp:function:: template <KokkosView SendView, KokkosView RecvView> void allgather(const SendView &sv, const RecvView &rv, MPI_Comm comm)

   Performs an allgather operation, gathering data from all processes and distributing it to all processes.

   :tparam SendView: The type of the view to be sent.
   :tparam RecvView: The type of the view to be received.
   :param sv: The view to be sent.
   :param rv: The view to be received.
   :param comm: The MPI communicator.

   If ``sv`` is a rank-0 view, the value from the jth rank will be placed in index j of ``rv``.

.. cpp:function:: template <KokkosView RecvView> void allgather(const RecvView &rv, MPI_Comm comm)

   Performs an in-place allgather operation, gathering data from all processes and distributing it to all processes.

   :tparam RecvView: The type of the view to be received.
   :param rv: The view to be received.
   :param comm: The MPI communicator.

.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView> void allgather(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Comm comm)

   Performs an allgather operation with a specified execution space, gathering data from all processes and distributing it to all processes.

   :tparam ExecSpace: The execution space.
   :tparam SendView: The type of the view to be sent.
   :tparam RecvView: The type of the view to be received.
   :param space: The execution space.
   :param sv: The view to be sent.
   :param rv: The view to be received.
   :param comm: The MPI communicator.

.. cpp:function:: inline void barrier(MPI_Comm comm)

   Blocks until all processes in the communicator have reached this routine.

   :param comm: The MPI communicator.


.. cpp:function:: template <KokkosView SendView, KokkosView RecvView> void reduce(const SendView &sv, const RecvView &rv, MPI_Op op, int root, MPI_Comm comm)

   Performs a reduction operation, combining data from all processes and distributing the result to the root process.

   :tparam SendView: The type of the view to be sent.
   :tparam RecvView: The type of the view to be received.
   :param sv: The view to be sent.
   :param rv: The view to be received.
   :param op: The MPI operation to be applied.
   :param root: The rank of the root process.
   :param comm: The MPI communicator.

.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView> void reduce(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Op op, int root, MPI_Comm comm)

   Performs a reduction operation with a specified execution space, combining data from all processes and distributing the result to the root process.

   :tparam ExecSpace: The execution space.
   :tparam SendView: The type of the view to be sent.
   :tparam RecvView: The type of the view to be received.
   :param space: The execution space.
   :param sv: The view to be sent.
   :param rv: The view to be received.
   :param op: The MPI operation to be applied.
   :param root: The rank of the root process.
   :param comm: The MPI communicator.


Related Types
-------------

.. cpp:namespace:: KokkosComm::mpi

.. _CommMode:

.. cpp:enum-class:: CommMode

    A scoped enum to specify the mode of an operation. Buffered mode is not supported.

    .. cpp:enumerator:: Standard

      Standard mode: the MPI implementation decides whether outgoing messages will be buffered. Send operations can be started whether or not a matching receive has been started. They may complete before a matching receive is started. Standard mode is non-local: successful completion of the send operation may depend on the occurrence of a matching receive.

    .. cpp:enumerator:: Ready

      Ready mode: Send operations may be started only if the matching receive is already started.

    .. cpp:enumerator:: Synchronous

      Synchronous mode: Send operations complete successfully only if a matching receive is started, and the receive operation has started to receive the message sent.

    .. cpp:enumerator:: Default

      Default mode is an alias for ``Standard`` mode, but lets users override the behavior of operations at compile-time using the ``KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE`` pre-processor define. This forces ``Synchronous`` mode for all "default-mode" operations, which can be useful for debugging purposes, e.g., for asserting that the communication scheme is correct.


