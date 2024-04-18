Core
====

.. list-table:: MPI API Support
    :widths: 20 30 15
    :header-rows: 1

    * - MPI
      - ``KokkosComm::``
      - ``Kokkos::View``
    * - ``MPI_Send``
      - ``send`` or ``send<CommMode::Standard>``
      - ✓
    * - ``MPI_Rsend``
      - ``send<CommMode::Ready>``
      - ✓
    * - MPI_Recv
      - recv
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
    * - MPI_Reduce
      - reduce
      - ✓
      - ✓

Point-to-point
--------------

.. cpp:function:: template <KokkosComm::CommMode SendMode, KokkosExecutionSpace ExecSpace, KokkosView SendView> \
                  Req KokkosComm::isend(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm)

    Wrapper for ``MPI_Isend``, ``MPI_Irsend`` and ``MPI_Issend``.

    :param space: The execution space to operate in
    :param sv: The data to send
    :param dest: the destination rank
    :param tag: the MPI tag
    :param comm: the MPI communicator
    :tparam SendView: A Kokkos::View to send
    :tparam ExecSpace: A Kokkos execution space to operate in
    :returns: A KokkosComm::Req representing the asynchronous communication and any lifetime-extended views.
    :tparam SendMode: A communication mode to use. If unspecified, defaults to a standard ``MPI_Isend``.
    :tparam SendMode: A CommMode_ to use. If unspecified, defaults to a standard ``MPI_Isend``.

    :tparam SendMode: A CommMode_ to use. If unspecified, defaults to a synchronous ``MPI_Issend`` if ``KokkosComm_FORCE_SYNCHRONOUS_MODE`` is enabled, otherwise defaults to a standard ``MPI_Isend``.
.. cpp:function:: template <KokkosComm::CommMode SendMode, KokkosExecutionSpace ExecSpace, KokkosView SendView> \
                  void KokkosComm::send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm)

    Wrapper for ``MPI_Send``, ``MPI_Rsend`` and ``MPI_Ssend``.

    :param space: The execution space to operate in
    :param sv: The data to send
    :param dest: the destination rank
    :param tag: the MPI tag
    :param comm: the MPI communicator
    :tparam SendView: A Kokkos::View to send
    :tparam ExecSpace: A Kokkos execution space to operate in
    :tparam SendMode: A communication mode to use. If unspecified, defaults to a standard ``MPI_Send``.
    :tparam SendMode: A CommMode_ to use. If unspecified, defaults to a standard ``MPI_Send``.
    :tparam SendMode: A CommMode_ to use. If unspecified, defaults to a synchronous ``MPI_Ssend`` if ``KokkosComm_FORCE_SYNCHRONOUS_MODE`` is enabled, otherwise defaults to a standard ``MPI_Send``.

.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView RecvView> \
                  void KokkosComm::recv(const ExecSpace &space, RecvView &rv, int src, int tag, MPI_Comm comm)

    MPI_Recv wrapper

    :param space: The execution space to operate in
    :param srv: The data to recv
    :param src: the source rank
    :param tag: the MPI tag
    :param comm: the MPI communicator
    :tparam Recv: A Kokkos::View to send
    :tparam ExecSpace: A Kokkos execution space to operate in


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

Related Types
-------------

.. _CommMode:

.. cpp:enum-class:: KokkosComm::CommMode

    A scoped enum to specify the mode of an operation. Buffered mode is not supported.

    .. cpp:enumerator:: KokkosComm::CommMode::Standard

      Standard mode: the MPI implementation decides whether outgoing messages will be buffered. Send operations can be started whether or not a matching receive has been started. They may complete before a matching receive is started. Standard mode is non-local: successful completion of the send operation may depend on the occurrence of a matching receive.

    .. cpp:enumerator:: KokkosComm::CommMode::Ready

      Ready mode: Send operations may be started only if the matching receive is already started.

    .. cpp:enumerator:: KokkosComm::CommMode::Synchronous

      Synchronous mode: Send operations complete successfully only if a matching receive is started, and the receive operation has started to receive themessage sent.

    .. cpp:enumerator:: KokkosComm::CommMode::Default

    Default mode is an alias for ``Standard`` mode, but lets users override the behavior of send operations at compile-time using the ``KOKKOSCOMM_FORCE_SYNCHRONOUS_SEND_MODE`` environment variable (pre-processor defined). This forces ``Synchronous`` mode for all "default-mode" send operations, which can be useful for debugging purposes, e.g., for asserting that the communication scheme is correct.


.. cpp:class:: KokkosComm::Req

    A wrapper around an MPI_Request that can also extend the lifetime of Views.

    .. cpp:function:: MPI_Request &KokkosComm::Req::mpi_req()

        Retrieve a reference to the held MPI_Request.

    .. cpp:function:: void KokkosComm::Req::wait()

        Call MPI_Wait on the held MPI_Request and drop copies of any previous arguments to Req::keep_until_wait().

    .. cpp:function:: template<typename View> \
                      void KokkosComm::Req::keep_until_wait(const View &v)

        Extend the lifetime of v at least until Req::wait() is called.
        This is useful to prevent a View from being destroyed during an asynchronous MPI operation.
