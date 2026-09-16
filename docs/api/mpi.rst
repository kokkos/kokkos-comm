************************
Low-level MPI interfaces
************************

.. list-table:: MPI API Support
    :widths: 40 50 30
    :header-rows: 1

    * - MPI routines
      - ``KokkosComm::mpi::`` namespace
      - ``Kokkos::View`` support
    * - ``MPI_Send``
      - ``send`` or ``send(... , CommModeStandard{})``
      - ✓
    * - ``MPI_Rsend``
      - ``send(... , CommModeReady{})``
      - ✓
    * - ``MPI_Ssend``
      - ``send(... , CommModeSynchronous{})``
      - ✓
    * - ``MPI_Isend``
      - ``isend`` or ``isend(... , CommModeStandard{})``
      - ✓
    * - ``MPI_Irsend``
      - ``isend(... , CommModeReady{})``
      - ✓
    * - ``MPI_Issend``
      - ``isend(... , CommModeSynchronous{})``
      - ✓
    * - ``MPI_Recv``
      - ``recv``
      - ✓
    * - ``MPI_Allgather``
      - ``allgather``
      - ✓
    * - ``MPI_Allgather`` (in-place)
      - ``allgather``
      - ✓
    * - ``MPI_Iallgather``
      - ``iallgather``
      - ✓
    * - ``MPI_Reduce``
      - ``reduce``
      - ✓
    * - ``MPI_Ireduce``
      - ``ireduce``
      - ✓
    * - ``MPI_Bcast``
      - ``broadcast``
      - ✓
    * - ``MPI_Ibcast``
      - ``ibroadcast``
      - ✓
    * - ``MPI_Alltoall``
      - ``alltoall``
      - ✓
    * - ``MPI_Ialltoall``
      - ``ialltoall``
      - ✓
    * - ``MPI_Allreduce``
      - ``allreduce``
      - ✓
    * - ``MPI_Iallreduce``
      - ``iallreduce``
      - ✓
    * - ``MPI_Scan``
      - ``inclusive_scan``
      - ✓
    * - ``MPI_Exscan``
      - ``exclusive_scan``
      - ✓
    * - ``MPI_Barrier``
      - ``barrier``
      - ✓


Point-to-point
==============

.. cpp:namespace:: KokkosComm::mpi

.. cpp:function:: template <KokkosView SendView> \
                  auto send(const SendView &sv, int dest, int tag, MPI_Comm comm) -> void

    Initiates a blocking send operation.

    :tparam SendView: The type of the view to be sent.

    :param sv: The view to be sent.
    :param dest: The destination rank.
    :param tag: The message tag.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView> \
                  auto send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) -> void

    Initiates a blocking send operation with a specified execution space.
    Uses ``DefaultCommMode``.

    :tparam ExecSpace: The execution space.
    :tparam SendView: The type of the view to be sent.

    :param space: The execution space.
    :param sv: The view to be sent.
    :param dest: The destination rank.
    :param tag: The message tag.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, CommunicationMode SendMode> \
                  auto send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm, SendMode) -> void

    Initiates a blocking send operation with a specified execution space and communication mode.
    The communication mode is selected by passing an instance of ``CommModeStandard``, ``CommModeReady``, or ``CommModeSynchronous`` as the last argument.

    :tparam ExecSpace: The execution space.
    :tparam SendView: The type of the view to be sent.
    :tparam SendMode: The communication mode type (e.g. ``CommModeStandard``, ``CommModeReady``, ``CommModeSynchronous``).

    :param space: The execution space.
    :param sv: The view to be sent.
    :param dest: The destination rank.
    :param tag: The message tag.
    :param comm: The MPI communicator.
    :param mode: A tag instance selecting the communication mode.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, CommunicationMode SendMode> \
                  auto isend(Communicator<MpiSpace, ExecSpace> &h, const SendView &sv, int dest, int tag, SendMode) -> Request<MpiSpace>

    Initiates a non-blocking send operation.
    The communication mode is selected by passing an instance of ``CommModeStandard``, ``CommModeReady``, or ``CommModeSynchronous`` as the last argument.

    :tparam ExecSpace: The execution space.
    :tparam SendView: The type of the view to be sent.
    :tparam SendMode: The communication mode type (e.g. ``CommModeStandard``, ``CommModeReady``, ``CommModeSynchronous``).

    :param h: The handle for the execution space and MPI.
    :param sv: The view to be sent.
    :param dest: The destination rank.
    :param tag: The message tag.
    :param mode: A tag instance selecting the communication mode.

    :return: A request object for the non-blocking send operation.


.. cpp:function:: template <KokkosView RecvView> \
                  auto recv(const RecvView &rv, int src, int tag, MPI_Comm comm, MPI_Status *status) -> void

    Initiates a blocking receive operation.

    :tparam RecvView: The type of the view to be received.

    :param rv: The view to be received.
    :param src: The source rank.
    :param tag: The message tag.
    :param comm: The MPI communicator.
    :param status: The MPI status object for the blocking receive operation.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView RecvView> \
                  auto recv(const ExecSpace &space, RecvView &rv, int src, int tag, MPI_Comm comm) -> void

    Initiates a blocking receive operation with a specified execution space.

    :tparam ExecSpace: The execution space.
    :tparam RecvView: The type of the view to be received.

    :param space: The execution space.
    :param rv: The view to be received.
    :param src: The source rank.
    :param tag: The message tag.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosView RecvView> \
                  auto irecv(const RecvView &rv, int src, int tag, MPI_Comm comm, MPI_Request &req) -> void

    Initiates a non-blocking receive operation.

    :tparam RecvView: The type of the view to be received.

    :param rv: The view to be received.
    :param src: The source rank.
    :param tag: The message tag.
    :param comm: The MPI communicator.
    :param req: The MPI request object for the non-blocking receive operation.

    :throws std::runtime_error: If the view is not contiguous.


.. cpp:function:: template <KokkosView SendView> \
                  auto isend(const SendView &sv, int dest, int tag, MPI_Comm comm, MPI_Request &req) -> void

    ``MPI_Isend`` with a ``Kokkos::View``.

    :tparam SendView: The type of the view to be sent.

    :param sv: The view to be sent (must be contiguous).
    :param dest: The destination rank.
    :param tag: The message tag.
    :param comm: The MPI communicator.
    :param req: The MPI request.


Collectives
===========

.. cpp:function:: template <KokkosView SendView, KokkosView RecvView> \
                  auto allgather(const SendView &sv, const RecvView &rv, MPI_Comm comm) -> void

    Performs an allgather operation, gathering data from all processes and distributing it to all processes.

    :tparam SendView: The type of the view to be sent.
    :tparam RecvView: The type of the view to be received.

    :param sv: The view to be sent.
    :param rv: The view to be received.
    :param comm: The MPI communicator.

    .. note: If ``sv`` is a rank-0 view, the value from the j-th rank will be placed in index j of ``rv``.


.. cpp:function:: template <KokkosView RecvView> \
                  auto allgather(const RecvView &rv, MPI_Comm comm) -> void

    Performs an in-place allgather operation, gathering data from all processes and distributing it to all processes.

    :tparam RecvView: The type of the view to be received.

    :param rv: The view to be received.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView> \
                  auto allgather(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Comm comm) -> void

    Performs an allgather operation with a specified execution space, gathering data from all processes and distributing it to all processes.

    :tparam ExecSpace: The execution space.
    :tparam SendView: The type of the view to be sent.
    :tparam RecvView: The type of the view to be received.

    :param space: The execution space.
    :param sv: The view to be sent.
    :param rv: The view to be received.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SView, KokkosView RView> \
                  auto iallgather(const ExecSpace &space, const SView sv, RView rv, MPI_Comm comm) -> Request<MpiSpace>

    ``MPI_Iallgather`` with ``Kokkos::View`` arguments.

    :tparam ExecSpace: The execution space.
    :tparam SView: The type of the view to be sent.
    :tparam RView: The type of the view to be received.

    :param space: The execution space.
    :param sv: The view to be sent (must be contiguous).
    :param rv: The view to be received (must be contiguous).
    :param comm: The MPI communicator.

    :return: A request object for the non-blocking all-gather.


.. cpp:function:: template <KokkosView SendView, KokkosView RecvView> \
                  auto reduce(const SendView &sv, const RecvView &rv, MPI_Op op, int root, MPI_Comm comm) -> void

    Performs a reduction operation, combining data from all processes and distributing the result to the root process.

    :tparam SendView: The type of the view to be sent.
    :tparam RecvView: The type of the view to be received.

    :param sv: The view to be sent.
    :param rv: The view to be received.
    :param op: The MPI operation to be applied.
    :param root: The rank of the root process.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView> \
                  auto reduce(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Op op, int root, MPI_Comm comm) -> void

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


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SView, KokkosView RView> \
                  auto ireduce(const ExecSpace &space, const SView &sv, RView &rv, MPI_Op op, int root, MPI_Comm comm) -> Request<MpiSpace>

    ``MPI_Ireduce`` with ``Kokkos::View`` arguments.

    :tparam ExecSpace: The execution space.
    :tparam SView: The type of the view to be sent.
    :tparam RView: The type of the view to be received.

    :param space: The execution space.
    :param sv: The view to be sent.
    :param rv: The view to be received (valid only on ``root``).
    :param op: The MPI operation to be applied.
    :param root: The rank of the root process.
    :param comm: The MPI communicator.

    :return: A request object for the non-blocking reduction.


.. cpp:function:: template <KokkosView View> \
                  auto broadcast(const View &v, int root, MPI_Comm comm) -> void

    ``MPI_Bcast`` with a ``Kokkos::View``.

    :tparam View: The type of the view to be broadcast.

    :param v: The view to be broadcast (must be contiguous).
    :param root: The rank of the root process.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView View> \
                  auto broadcast(const ExecSpace &space, const View &v, int root, MPI_Comm comm) -> void

    ``MPI_Bcast`` with a ``Kokkos::View`` and execution space.

    :tparam ExecSpace: The execution space.
    :tparam View: The type of the view to be broadcast.

    :param space: The execution space.
    :param v: The view to be broadcast (must be contiguous).
    :param root: The rank of the root process.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView View> \
                  auto ibroadcast(const ExecSpace &space, View &v, int root, MPI_Comm comm) -> Request<MpiSpace>

    ``MPI_Ibcast`` with a ``Kokkos::View``.

    :tparam ExecSpace: The execution space.
    :tparam View: The type of the view to be broadcast.

    :param space: The execution space.
    :param v: The view to be broadcast (must be contiguous).
    :param root: The rank of the root process.
    :param comm: The MPI communicator.

    :return: A request object for the non-blocking broadcast.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView> \
                  auto alltoall(const ExecSpace &space, const SendView &sv, const size_t sendCount, const RecvView &rv, const size_t recvCount, MPI_Comm comm) -> void

    ``MPI_Alltoall`` with ``Kokkos::View`` arguments.

    :tparam ExecSpace: The execution space.
    :tparam SendView: The type of the view to be sent.
    :tparam RecvView: The type of the view to be received.

    :param space: The execution space.
    :param sv: The view to be sent (must be contiguous).
    :param sendCount: The number of elements to send to each process.
    :param rv: The view to be received (must be contiguous).
    :param recvCount: The number of elements to receive from each process.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView RecvView> \
                  auto alltoall(const ExecSpace &space, const RecvView &rv, const size_t recvCount, MPI_Comm comm) -> void

    ``MPI_Alltoall`` (in-place) with a ``Kokkos::View``.

    :tparam ExecSpace: The execution space.
    :tparam RecvView: The type of the view to be received.

    :param space: The execution space.
    :param rv: The view to be received (must be contiguous).
    :param recvCount: The number of elements to receive from each process.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SView, KokkosView RView> \
                  auto ialltoall(const ExecSpace &space, const SView sv, RView rv, int count, MPI_Comm comm) -> Request<MpiSpace>

    ``MPI_Ialltoall`` with ``Kokkos::View`` arguments.

    :tparam ExecSpace: The execution space.
    :tparam SView: The type of the view to be sent.
    :tparam RView: The type of the view to be received.

    :param space: The execution space.
    :param sv: The view to be sent (must be contiguous).
    :param rv: The view to be received (must be contiguous).
    :param count: The number of elements sent to (and received from) each process.
    :param comm: The MPI communicator.

    :return: A request object for the non-blocking all-to-all.


.. cpp:function:: template <KokkosView SendView, KokkosView RecvView> \
                  auto allreduce(const SendView &sv, const RecvView &rv, MPI_Op op, MPI_Comm comm) -> void

    ``MPI_Allreduce`` with ``Kokkos::View`` arguments.

    :tparam SendView: The type of the view to be sent.
    :tparam RecvView: The type of the view to be received.

    :param sv: The view to be sent (must be contiguous).
    :param rv: The view to be received (must be contiguous, same size as ``sv``).
    :param op: The MPI operation to be applied.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosView View> \
                  auto allreduce(const View &v, MPI_Op op, MPI_Comm comm) -> void

    ``MPI_Allreduce`` (in-place) with a ``Kokkos::View``.

    :tparam View: The type of the view to be reduced.

    :param v: The view to be reduced (must be contiguous).
    :param op: The MPI operation to be applied.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView> \
                  auto allreduce(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Op op, MPI_Comm comm) -> void

    ``MPI_Allreduce`` with ``Kokkos::View`` arguments and an execution space.

    :tparam ExecSpace: The execution space.
    :tparam SendView: The type of the view to be sent.
    :tparam RecvView: The type of the view to be received.

    :param space: The execution space.
    :param sv: The view to be sent (must be contiguous).
    :param rv: The view to be received (must be contiguous, same size as ``sv``).
    :param op: The MPI operation to be applied.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView View> \
                  auto allreduce(const ExecSpace &space, const View &v, MPI_Op op, MPI_Comm comm) -> void

    ``MPI_Allreduce`` (in-place) with a ``Kokkos::View`` and an execution space.

    :tparam ExecSpace: The execution space.
    :tparam View: The type of the view to be reduced.

    :param space: The execution space.
    :param v: The view to be reduced (must be contiguous).
    :param op: The MPI operation to be applied.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosView SView, KokkosView RView, KokkosExecutionSpace ExecSpace> \
                  auto iallreduce(const ExecSpace &space, const SView sv, RView rv, MPI_Op op, MPI_Comm comm) -> Request<MpiSpace>

    ``MPI_Iallreduce`` with ``Kokkos::View`` arguments.

    :tparam SView: The type of the view to be sent.
    :tparam RView: The type of the view to be received.
    :tparam ExecSpace: The execution space.

    :param space: The execution space.
    :param sv: The view to be sent (must be contiguous).
    :param rv: The view to be received (must be contiguous, same size as ``sv``).
    :param op: The MPI operation to be applied.
    :param comm: The MPI communicator.

    :return: A request object for the non-blocking all-reduce.


.. cpp:function:: template <KokkosView SendView, KokkosView RecvView> \
                  auto inclusive_scan(const SendView &sv, const RecvView &rv, MPI_Op op, MPI_Comm comm) -> void

    ``MPI_Scan`` with ``Kokkos::View`` arguments.

    :tparam SendView: The type of the view to be sent.
    :tparam RecvView: The type of the view to be received.

    :param sv: The view to be sent (must be contiguous, rank ≤ 1).
    :param rv: The view to be received (must be contiguous, same size as ``sv``).
    :param op: The MPI operation to be applied.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView> \
                  auto inclusive_scan(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Op op, MPI_Comm comm) -> void

    ``MPI_Scan`` with ``Kokkos::View`` arguments and an execution space.

    :tparam ExecSpace: The execution space.
    :tparam SendView: The type of the view to be sent.
    :tparam RecvView: The type of the view to be received.

    :param space: The execution space.
    :param sv: The view to be sent (must be contiguous).
    :param rv: The view to be received (must be contiguous, same size as ``sv``).
    :param op: The MPI operation to be applied.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosView SendView, KokkosView RecvView> \
                  auto exclusive_scan(const SendView &sv, const RecvView &rv, MPI_Op op, MPI_Comm comm) -> void

    ``MPI_Exscan`` with ``Kokkos::View`` arguments.

    :tparam SendView: The type of the view to be sent.
    :tparam RecvView: The type of the view to be received.

    :param sv: The view to be sent (must be contiguous, rank ≤ 1).
    :param rv: The view to be received (must be contiguous, same size as ``sv``).
    :param op: The MPI operation to be applied.
    :param comm: The MPI communicator.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView> \
                  auto exclusive_scan(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Op op, MPI_Comm comm) -> void

    ``MPI_Exscan`` with ``Kokkos::View`` arguments and an execution space.

    :tparam ExecSpace: The execution space.
    :tparam SendView: The type of the view to be sent.
    :tparam RecvView: The type of the view to be received.

    :param space: The execution space.
    :param sv: The view to be sent (must be contiguous).
    :param rv: The view to be received (must be contiguous, same size as ``sv``).
    :param op: The MPI operation to be applied.
    :param comm: The MPI communicator.


.. cpp:function:: inline auto barrier(MPI_Comm comm) -> void

    Blocks until all processes in the communicator have reached this routine.

    :param comm: The MPI communicator.


Related Types
=============

.. cpp:namespace:: KokkosComm::mpi

.. _CommModeStandard:

.. cpp:struct:: CommModeStandard

    Tag type for MPI standard mode. The MPI implementation decides whether outgoing messages will be buffered. Send operations can be started whether or not a matching receive has been started. They may complete before a matching receive is started. Standard mode is non-local: successful completion of the send operation may depend on the occurrence of a matching receive.

.. _CommModeReady:

.. cpp:struct:: CommModeReady

    Tag type for MPI ready mode. Send operations may be started only if the matching receive is already started.

.. _CommModeSynchronous:

.. cpp:struct:: CommModeSynchronous

    Tag type for MPI synchronous mode. Send operations complete successfully only if a matching receive is started, and the receive operation has started to receive the message sent.

.. _DefaultCommMode:

.. cpp:type:: DefaultCommMode = CommModeStandard

    A type alias for the default communication mode. Defaults to ``CommModeStandard``, but when ``KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE`` is defined, it aliases ``CommModeSynchronous`` instead. This allows users to force all default-mode operations into synchronous mode at compile time, which can be useful for debugging and asserting that the communication scheme is correct.

.. _CommunicationMode:

.. cpp:concept:: template <typename T> \
               CommunicationMode

    A concept satisfied by the communication mode tag types: ``CommModeStandard``, ``CommModeReady``, and ``CommModeSynchronous``.
