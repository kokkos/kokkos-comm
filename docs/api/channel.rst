*******
Channel
*******

.. cpp:namespace:: KokkosComm

.. cpp:class:: template <CommunicationSpace CommSpace = MpiSpace> Channel

    An MPI persistent communication channel for repeated point-to-point exchanges.

    Only the ``Channel<MpiSpace>`` specialization is currently provided.

    ``Channel`` binds a fixed source rank, destination rank, and message tag.
    After registering send and receive buffers with :cpp:func:`sendinit` and :cpp:func:`recvinit`, the user calls :cpp:func:`start` to launch all registered operations and :cpp:func:`wait` to block until they complete.

    Each registered operation is represented by a persistent MPI request.
    Register the communication pattern once, then repeat ``start`` / ``wait`` cycles to execute the same operations.
    The persistent requests are released when the channel is destroyed.

    A persistent request retains the buffer address, element count, and datatype supplied during registration.
    ``Channel`` does not retain the View object or its allocation.
    The application must therefore keep every registered buffer allocation valid and at the same address until the channel is destroyed.
    Between a completed :cpp:func:`wait` and the next :cpp:func:`start`, send buffers may be modified and receive buffers may be read or modified.
    While an operation is active, the usual MPI buffer-access restrictions apply.

    The communicator is borrowed rather than duplicated.
    It must remain valid until the channel is destroyed, and the channel must be destroyed before ``MPI_Finalize``.
    Every call to :cpp:func:`start` must be matched by a successful :cpp:func:`wait` before destruction.

    :tparam CommSpace: The communication backend. Only ``MpiSpace`` is supported.

    .. cpp:function:: explicit Channel(int dest_rank, int src_rank, int tag, MPI_Comm comm)

        Constructs a ``Channel`` with fixed endpoints.

        :param dest_rank: The destination rank for send operations.
        :param src_rank: The source rank for receive operations.
        :param tag: The message tag.
        :param comm: The borrowed MPI communicator used by every registered operation.

    .. cpp:function:: ~Channel()

        Releases all registered persistent requests with ``MPI_Request_free``.
        The channel must have no active operations; the destructor does not wait for them.

    .. cpp:function:: Channel(const Channel&) = delete
                      auto operator=(const Channel&) -> Channel& = delete

        Copy construction and copy assignment are deleted because a channel exclusively owns its persistent requests.

    .. cpp:function:: Channel(Channel&& other) noexcept

        Transfers ownership of all persistent requests and the borrowed communicator from ``other``.
        The moved-from channel is empty and may be safely destroyed.

    .. cpp:function:: auto operator=(Channel&& other) noexcept -> Channel&

        Releases the destination's existing persistent requests, then transfers ownership from ``other``.
        The moved-from channel is empty and may be safely destroyed.

        The destination must have no active operations before assignment.

    .. cpp:function:: template <class SendView> void sendinit(SendView view)

        Registers a send buffer with the channel.

        The View's data type determines the element type communicated.
        Its underlying allocation must remain valid until the channel is destroyed.

        Each call registers an additional persistent send operation.

        :tparam SendView: A Kokkos View type.
        :param view: The view to register as a send buffer.

    .. cpp:function:: template <class RecvView> void recvinit(RecvView view)

        Registers a receive buffer with the channel.

        The View's data type determines the element type communicated.
        Its underlying allocation must remain valid until the channel is destroyed.

        Each call registers an additional persistent receive operation.

        :tparam RecvView: A Kokkos View type.
        :param view: The view to register as a receive buffer.

    .. cpp:function:: void start()

        Activates all registered persistent requests.

        All pending Kokkos kernel work is fenced before the communication operations are initiated, ensuring that send buffers are fully populated and receive buffers are not in use.

    .. cpp:function:: void wait()

        Blocks until all active persistent requests have completed.

        Upon return, the persistent MPI requests become inactive but remain registered.
        Call :cpp:func:`start` again to repeat the same communication operations.
