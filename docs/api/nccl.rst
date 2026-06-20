*************************
Low-level NCCL interfaces
*************************

.. list-table:: NCCL API Support
    :widths: 40 50 30
    :header-rows: 1

    * - NCCL routines
      - ``KokkosComm::Experimental::nccl::`` namespace
      - ``Kokkos::View`` support
    * - ``ncclSend``
      - ``send``
      - ✓
    * - ``ncclRecv``
      - ``recv``
      - ✓
    * - ``ncclAllGather``
      - ``allgather``
      - ✓
    * - ``ncclAllReduce``
      - ``allreduce``
      - ✓
    * - ``ncclReduce``
      - ``reduce``
      - ✓
    * - ``ncclBroadcast``
      - ``broadcast``
      - ✓
    * - ``ncclAllToAll``
      - ``alltoall``
      - ✓


Point-to-point
==============

.. cpp:namespace:: KokkosComm::Experimental::nccl

.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView> \
                  auto send(const ExecSpace &space, const SendView &sv, int peer, ncclComm_t comm) -> Request<NcclSpace>

    Initiates a non-blocking send operation on the given CUDA stream.

    :tparam ExecSpace: The execution space (e.g. ``Kokkos::Cuda``).
    :tparam SendView: The type of the view to be sent.

    :param space: The execution space instance.
    :param sv: The view to be sent.
    :param peer: The destination rank.
    :param comm: The NCCL communicator.

    :return: A request object representing the asynchronous send operation.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView RecvView> \
                  auto recv(const ExecSpace &space, RecvView &rv, int peer, ncclComm_t comm) -> Request<NcclSpace>

    Initiates a non-blocking receive operation on the given CUDA stream.

    :tparam ExecSpace: The execution space (e.g. ``Kokkos::Cuda``).
    :tparam RecvView: The type of the view to be received.

    :param space: The execution space instance.
    :param rv: The view to be received.
    :param peer: The source rank.
    :param comm: The NCCL communicator.

    :return: A request object representing the asynchronous receive operation.


Collectives
===========

.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView> \
                  auto allgather(const ExecSpace &space, const SendView &sv, const RecvView &rv, ncclComm_t comm) -> Request<NcclSpace>

    Performs an all-gather operation, gathering data from all processes and distributing it to all processes.

    :tparam ExecSpace: The execution space (e.g. ``Kokkos::Cuda``).
    :tparam SendView: The type of the view to be sent.
    :tparam RecvView: The type of the view to be received.

    :param space: The execution space instance.
    :param sv: The view to be sent.
    :param rv: The view to be received.
    :param comm: The NCCL communicator.

    :return: A request object representing the asynchronous all-gather operation.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView> \
                  auto allreduce(const ExecSpace &space, const SendView &sv, const RecvView &rv, ncclRedOp_t op, ncclComm_t comm) -> Request<NcclSpace>

    Performs an all-reduce operation, combining data from all processes and distributing the result to all processes.

    :tparam ExecSpace: The execution space (e.g. ``Kokkos::Cuda``).
    :tparam SendView: The type of the view to be sent.
    :tparam RecvView: The type of the view to be received.

    :param space: The execution space instance.
    :param sv: The view to be sent.
    :param rv: The view to be received.
    :param op: The NCCL reduction operation to be applied.
    :param comm: The NCCL communicator.

    :return: A request object representing the asynchronous all-reduce operation.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView> \
                  auto reduce(const ExecSpace &space, const SendView &sv, RecvView &rv, ncclRedOp_t op, int root, int rank, ncclComm_t comm) -> Request<NcclSpace>

    Performs a reduction operation, combining data from all processes and placing the result on the root process.

    :tparam ExecSpace: The execution space (e.g. ``Kokkos::Cuda``).
    :tparam SendView: The type of the view to be sent.
    :tparam RecvView: The type of the view to be received.

    :param space: The execution space instance.
    :param sv: The view to be sent.
    :param rv: The view to be received (used on the root process).
    :param op: The NCCL reduction operation to be applied.
    :param root: The rank of the root process.
    :param rank: The rank of the calling process.
    :param comm: The NCCL communicator.

    :return: A request object representing the asynchronous reduce operation.


.. cpp:function:: template <KokkosView View> \
                  auto broadcast(const Kokkos::Cuda &space, View &v, int root, ncclComm_t comm) -> Request<NcclSpace>

    Broadcasts data from the root process to all other processes in the communicator.

    :tparam View: The type of the view to be broadcast.

    :param space: The ``Kokkos::Cuda`` execution space instance.
    :param v: The view to be broadcast (in-place on all ranks).
    :param root: The rank of the root process.
    :param comm: The NCCL communicator.

    :return: A request object representing the asynchronous broadcast operation.


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView> \
                  auto alltoall(const ExecSpace &space, const SendView &sv, const RecvView &rv, int count, ncclComm_t comm) -> Request<NcclSpace>

    Performs an all-to-all exchange where each process sends ``count`` elements to every other process.

    :tparam ExecSpace: The execution space (e.g. ``Kokkos::Cuda``).
    :tparam SendView: The type of the view to be sent.
    :tparam RecvView: The type of the view to be received.

    :param space: The execution space instance.
    :param sv: The view to be sent.
    :param rv: The view to be received.
    :param count: The number of elements sent to each process.
    :param comm: The NCCL communicator.

    :return: A request object representing the asynchronous all-to-all operation.
