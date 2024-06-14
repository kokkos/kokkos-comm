Core
****

MPI API Support
===============

.. list-table:: MPI API Support
    :widths: 40 30 15
    :header-rows: 1

    * - MPI
      - KokkosComm
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


Initialization and finalization
-------------------------------

KokkosComm provides a unified interface for initializing and finalizing both Kokkos and MPI.

.. Attention:: It is strongly recommended to use KokkosComm's initialization and finalization functions instead of their respective Kokkos and MPI counterparts. However, users have two options for using KokkosComm:

    1. Initialize/finalize KokkosComm using ``KokkosComm::{initialize,finalize}``. In this case, it is **forbidden** to call ``MPI_{Init,Init_thread,Finalize}`` and ``Kokkos::{initialize,finalize}`` (otherwise, the application will abort).
    2. Initialize/finalize MPI and Kokkos manually through their respective interfaces. In this case, it is **forbidden** to call ``KokkosComm::{initialize,finalize}`` (otherwise, the application will abort).

.. cpp:enum-class:: KokkosComm::ThreadSupportLevel

    A scoped enum to wrap the MPI thread support levels.

    .. cpp:enumerator:: KokkosComm::ThreadSupportLevel::Single

        Only one thread will execute.

    .. cpp:enumerator:: KokkosComm::ThreadSupportLevel::Funneled

        The process may be multi-threaded, but only the main thread will make MPI calls (all MPI calls are funneled to the main thread).

    .. cpp:enumerator:: KokkosComm::ThreadSupportLevel::Serialized

        The process may be multi-threaded, and multiple threads may make MPI calls, but only one at a time: MPI calls are not made concurrently from two distinct threads (all MPI calls are serialized).

    .. cpp:enumerator:: KokkosComm::ThreadSupportLevel::Multiple

        Multiple threads may call MPI, with no restrictions.


.. cpp:function:: void KokkosComm::initialize(int &argc, char *argv[], KokkosComm::ThreadSupportLevel required = KokkosComm::ThreadSupportLevel::Multiple)

    Initializes the MPI execution environment with the required MPI thread level support (``Multiple`` if left unspecified), and then initializes the Kokkos execution environment. This function also strips ``--kokkos-help`` flags to prevent Kokkos from printing them on all MPI ranks.

    :param argc: Non-negative value representing the number of command-line arguments passed to the program.
    :param argv: Pointer to the first element of an array of ``argc + 1`` pointers, of which the last one is null and the previous, if any, point to null-terminated multi-byte strings that represent the arguments passed to the program.
    :param required: Level of desired MPI thread support.

    **Requirements:**

    * ``KokkosComm::initialize`` has the same combined requirements as ``MPI_{Init,Init_thread}`` and ``Kokkos::initialize``.
    * ``KokkosComm::initialize`` must be called in place of ``MPI_Init`` and ``Kokkos::initialize``.
    * User-initiated MPI objects cannot be constructed, and MPI functions cannot be called until after ``KokkosComm::initialize`` is called.
    * User-initiated Kokkos objects cannot be constructed until after ``KokkosComm::initialize`` is called.

.. cpp:function:: void KokkosComm::finalize()

    Terminates the Kokkos and MPI execution environments.

    Programs are ill-formed if they do not call this function *after* calling ``KokkosComm::initialize``.

    **Requirements:**

    * ``KokkosComm::finalize`` has the same combined requirements as ``MPI_Finalize`` and ``Kokkos::finalize``.
    * ``KokkosComm::finalize`` must be called in place of ``MPI_Finalize`` and ``Kokkos::finalize``.
    * ``KokkosComm::finalize`` must be called after user-initialized Kokkos objects are out of scope.


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


.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView> \
                  void KokkosComm::allgather(const ExecSpace &space, const SendView &sv, const RecvView &rv, MPI_Comm comm)

    MPI_Allgather wrapper

    :param space: The execution space to operate in
    :param sv: The data to send
    :param rv: The view to receive into
    :param comm: the MPI communicator
    :tparam SendView: A Kokkos::View to send. Contiguous and rank less than 2.
    :tparam RecvView: A Kokkos::View to recv. Contiguous and rank 1.
    :tparam ExecSpace: A Kokkos execution space to operate in

    If ``sv`` is a rank-0 view, the value from the jth rank will be placed in index j of ``rv``.

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

    A wrapper around an MPI_Request that can also extend the lifetime of Views.

    .. cpp:function:: MPI_Request &KokkosComm::Req::mpi_req()

        Retrieve a reference to the held MPI_Request.

    .. cpp:function:: void KokkosComm::Req::wait()

        Call MPI_Wait on the held MPI_Request and drop copies of any previous arguments to Req::keep_until_wait().

    .. cpp:function:: template<typename View> \
                      void KokkosComm::Req::keep_until_wait(const View &v)

        Extend the lifetime of v at least until Req::wait() is called.
        This is useful to prevent a View from being destroyed during an asynchronous MPI operation.
