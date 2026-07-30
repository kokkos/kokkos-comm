*******
Channel
*******

.. cpp:namespace:: KokkosComm

.. cpp:class:: template <CommunicationSpace CommSpace = MpiSpace> Channel

    An MPI persistent communication channel for repeated point-to-point exchanges.

    Only the ``Channel<MpiSpace>`` specialization is currently provided.

    ``Channel`` binds a fixed source rank, destination rank, and message tag. After
    registering send and receive buffers with :cpp:func:`sendinit` and
    :cpp:func:`recvinit`, the user calls :cpp:func:`start` to launch all queued
    operations and :cpp:func:`wait` to block until they complete.

    The communication pattern is established once at construction time. Subsequent
    ``start`` / ``wait`` cycles reuse the same endpoints and message descriptors,
    which lets the communication backend optimize repeated exchanges. This is
    particularly useful for iterative algorithms where the same point-to-point
    pattern is repeated every iteration.

    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    .. cpp:function:: explicit Channel(int dest_rank, int src_rank, int tag, typename CommSpace::communicator_type comm)

        Constructs a ``Channel`` with fixed endpoints.

        :param dest_rank: The destination rank for send operations.
        :param src_rank: The source rank for receive operations.
        :param tag: The message tag.
        :param comm: A communicator handle for the target communication backend.

    .. cpp:function:: template <class SendView> void sendinit(SendView view)

        Registers a send buffer with the channel.

        The view's data type determines the element type communicated. The view must
        remain valid until the corresponding :cpp:func:`start` has been matched by a
        :cpp:func:`wait`.

        Multiple calls accumulate into the send queue.

        :tparam SendView: A Kokkos View type.
        :param view: The view to register as a send buffer.

    .. cpp:function:: template <class RecvView> void recvinit(RecvView view)

        Registers a receive buffer with the channel.

        The view's data type determines the element type communicated. The view must
        remain valid until the corresponding :cpp:func:`start` has been matched by a
        :cpp:func:`wait`.

        Multiple calls accumulate into the receive queue.

        :tparam RecvView: A Kokkos View type.
        :param view: The view to register as a receive buffer.

    .. cpp:function:: void start()

        Launches all queued send and receive operations.

        All pending Kokkos kernel work is fenced before the communication operations
        are initiated, ensuring that send buffers are fully populated and receive
        buffers are not in use.

    .. cpp:function:: void wait()

        Blocks until all previously started send and receive operations have completed.

        Upon return, the internal request queues are consumed. To perform another
        exchange with the same or different buffers, register new buffers with
        :cpp:func:`sendinit` / :cpp:func:`recvinit` and call :cpp:func:`start` again.
