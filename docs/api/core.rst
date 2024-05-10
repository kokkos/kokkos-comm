Core
====

.. list-table:: MPI API Support
    :widths: 40 30 15
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

.. cpp:function:: template <KokkosComm::CommMode SendMode, KokkosExecutionSpace ExecSpace, KokkosView SendView> \
                  Req KokkosComm::isend(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm)

    Wrapper for ``MPI_Isend``, ``MPI_Irsend`` and ``MPI_Issend``.
    The communication operation will be inserted into ``space``.
    The caller may safely call this function on data previously produced by operations in ``space`` without first fencing ```space```.

    .. warning::
        Even if ``space`` is fenced after the call to this function, the communication operation is not complete until the ``wait`` operation on the returned ``Req`` is called.

    :param space: The execution space to operate in
    :param sv: The data to send
    :param dest: the destination rank
    :param tag: the MPI tag
    :param comm: the MPI communicator
    :tparam SendMode: A CommMode_ to use. If unspecified, defaults to a synchronous ``MPI_Issend`` if ``KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE`` is defined, otherwise defaults to a standard ``MPI_Isend``.
    :tparam SendView: A Kokkos::View to send
    :tparam ExecSpace: A Kokkos execution space to operate in
    :returns: A KokkosComm::Req representing the asynchronous communication and any lifetime-extended views.

.. cpp:function:: template <KokkosComm::CommMode SendMode, KokkosExecutionSpace ExecSpace, KokkosView SendView> \
                  void KokkosComm::send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm)

    Wrapper for ``MPI_Send``, ``MPI_Rsend`` and ``MPI_Ssend``.

    :param space: The execution space to operate in
    :param sv: The data to send
    :param dest: the destination rank
    :param tag: the MPI tag
    :param comm: the MPI communicator
    :tparam SendMode: A CommMode_ to use. If unspecified, defaults to a synchronous ``MPI_Ssend`` if ``KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE`` is defined, otherwise defaults to a standard ``MPI_Send``.
    :tparam SendView: A Kokkos::View to send
    :tparam ExecSpace: A Kokkos execution space to operate in

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

      Synchronous mode: Send operations complete successfully only if a matching receive is started, and the receive operation has started to receive the message sent.

    .. cpp:enumerator:: KokkosComm::CommMode::Default

      Default mode is an alias for ``Standard`` mode, but lets users override the behavior of operations at compile-time using the ``KOKKOSCOMM_FORCE_SYNCHRONOUS_MODE`` pre-processor define. This forces ``Synchronous`` mode for all "default-mode" operations, which can be useful for debugging purposes, e.g., for asserting that the communication scheme is correct.


.. cpp:class:: KokkosComm::Req

    A communication handle representing an asychronous communication and an associated Kokkos execution space instance. The handle is scoped to the space instance used in the communication call that produced it. 



    .. cpp:function:: MPI_Request &KokkosComm::Req::mpi_req()

        Retrieve a reference to the held MPI_Request.

    .. cpp:function:: void KokkosComm::Req::wait()

        Require that the communication be completed before any further work can be exected in the associated execution space instance. May or may not fence. Consider the following example.

        .. code-block:: c++
          :linenos:

          using KC = KokkosComm;
          Kokkos::parallel_for(space, ...);
          auto req = KC::isend(space, ...); // isend 1
          Kokkos::parallel_for(space, ...); // runs concurrently with isend 1, does not touch send view
          req.wait();                       // blocks space until isend 1 is complete. May or may not fence.
          Kokkos::parallel_for(space, ...); // safe to overwrite the send buffer
          space.fence(); // wait for all to complete

        Here, ``parallel_for`` on line 6 can overwrite the send buffer because ``req.wait()`` means that isend 1 must be done before additional work can be done in ``space``. This MAY be achieved by an internal call to ``space.fence()``, but some other mechanism may be used. If the host thread wants to be sure that the communication is done, it must separately call ``space.fence()``.