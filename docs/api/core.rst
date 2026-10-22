****
Core
****

.. cpp:namespace:: KokkosComm

Data Structures
===============

Communicators
-------------

.. cpp:class:: template <CommunicationSpace Comm = DefaultCommunicationSpace, KokkosExecutionSpace Exec = Kokkos::DefaultExecutionSpace>\
               Communicator

    Template class for communicator wrappers of different communication space types.

    Communicators wrap a communication library-specific communicator (e.g. ``MPI_Comm``) and a Kokkos execution space, tightly coupling the two.

    ``Communicator`` objects are constructed via factory member functions. The parameterized constructor is private, and no default constructor is defined.
    They are move-only objects: copy construction and copy assignment are explicitly deleted. Use the ``duplicate`` member functions to create equivalent "copies" of communicators.
    There is always exactly one owner of a ``Communicator``.

    :tparam Comm: The communication space (transport backend) to use. Defaults to ``DefaultCommunicationSpace``.
    :tparam Exec: The Kokkos execution space to use. Defaults to ``Kokkos::DefaultExecutionSpace``.

    .. cpp:type:: execution_space = Exec
    .. cpp:type:: communication_space = Comm
    .. cpp:type:: communicator_type = Comm::communicator_type
    .. cpp:type:: size_type = Comm::size_type
    .. cpp:type:: rank_type = Comm::rank_type

Common interfaces
^^^^^^^^^^^^^^^^^

Both specializations share the following interface:

.. cpp:function:: ~Communicator() noexcept

    Destructor.

.. cpp:function:: Communicator(const Communicator&) = delete

    Copy constructor is deleted because a ``Communicator`` cannot be implicitly copied. Use ``duplicate`` instead.

.. cpp:function:: auto operator=(const Communicator&) -> Communicator& = delete

    Copy assignment operator is deleted because a ``Communicator`` cannot be implicitly copied. Use ``duplicate`` instead.

.. cpp:function:: Communicator(Communicator&&) noexcept

    Move-constructs a ``Communicator``.

.. cpp:function:: auto operator=(Communicator&&) noexcept -> Communicator&

    Move-assigns a ``Communicator``.

.. cpp:function:: [[nodiscard]] constexpr auto size() const noexcept -> size_type

    :returns: The size (i.e., number of processes) in the communicator.

.. cpp:function:: [[nodiscard]] constexpr auto rank() const noexcept -> rank_type

    :returns: The rank that identifies the calling process within the communicator.

.. cpp:function:: [[nodiscard]] auto split(int color, int key) noexcept -> std::optional<Communicator<Comm, Exec>>

    Splits a ``Communicator``.

    Given a color and a key, creates as many new communicators as distinct values of ``color`` are given, ordering processes according to the value of ``key``.
    All processes with the same color join the same communicator.

    :param color: A value controlling in which split communicator the calling process should be in.
    :param key: A value ordering the calling process within the split communicator.
    :returns: A communicator if the calling process is part of one of the split communicators, ``std::nullopt`` if the color is a special value excluding the process at this rank or on error.

.. cpp:function:: [[nodiscard]] auto duplicate() noexcept -> std::optional<Communicator<Comm, Exec>>

    Duplicates a ``Communicator``.

    :returns: A communicator on success, ``std::nullopt`` on error.

MPI specialization
^^^^^^^^^^^^^^^^^^

.. cpp:class:: template <KokkosExecutionSpace Exec> Communicator<MpiSpace, Exec>

    Communicator specialization for the :cpp:class:`MpiSpace` communication space.
    Wraps an ``MPI_Comm`` handle.

    .. cpp:type:: execution_space = Exec
    .. cpp:type:: communication_space = MpiSpace
    .. cpp:type:: communicator_type = MPI_Comm
    .. cpp:type:: size_type = int
    .. cpp:type:: rank_type = int

    .. cpp:function:: [[nodiscard]] static auto from_raw(MPI_Comm comm, const Exec& exec = Exec{}) noexcept -> Communicator<MpiSpace, Exec>

        Constructs a ``Communicator`` from a raw ``MPI_Comm`` handle and a Kokkos execution space instance. Defaults ``exec`` to ``Exec``.
        The passed handle must be a valid handle and must not be an inter-communicator parent handle.
        The returned communicator does not own the underlying handle, and the user is responsible for destroying it.

        :param comm: A valid communicator handle.
        :param exec: A Kokkos execution space instance. Defaults to ``Kokkos::DefaultExecutionSpace``.
        :returns: A communicator on success, ``std::nullopt`` if the passed handle was ``MPI_COMM_NULL``.

    .. cpp:function:: [[nodiscard]] static auto split_from_raw(const MPI_Comm comm, int color, int key, const Exec& exec = Exec{}) noexcept -> std::optional<Communicator<MpiSpace, Exec>>

        Splits from a raw MPI communicator and associates it to a Kokkos execution space instance. Defaults ``exec`` to ``Exec``.

        Creates as many new communicators as distinct values of ``color`` are given, and orders processes according to the value of ``key``. All processes with the same value of ``color`` join the same communicator.
        A process that passes ``MPI_UNDEFINED`` as ``color`` will not join a new communicator.

        :param comm: A valid communicator handle.
        :param color: A value controlling in which split communicator the calling process should be in.
        :param key: A value ordering the calling process within the split communicator.
        :param exec: A Kokkos execution space instance. Defaults to ``Kokkos::DefaultExecutionSpace``.
        :returns: A split communicator on success, ``std::nullopt`` if the passed color was ``MPI_UNDEFINED`` or on error.

    .. cpp:function:: [[nodiscard]] static auto duplicate_from_raw(const MPI_Comm comm, const Exec& exec = Exec{}) noexcept -> std::optional<Communicator<MpiSpace, Exec>>

        Duplicates from a raw MPI communicator.

        :param comm: A valid communicator handle.
        :param exec: A Kokkos execution space instance. Defaults to ``Kokkos::DefaultExecutionSpace``.
        :returns: A communicator on success, ``std::nullopt`` on error.

    .. cpp:function:: [[nodiscard]] auto comm() noexcept -> MPI_Comm&
                      [[nodiscard]] auto comm() const noexcept -> const MPI_Comm&

        :returns: A reference to the underlying ``MPI_Comm`` object.

    .. cpp:function:: [[nodiscard]] auto exec() const noexcept -> const execution_space&

        :returns: A const reference to the associated execution space instance.

NCCL specialization
^^^^^^^^^^^^^^^^^^^

.. cpp:class:: template <> Communicator<Experimental::NcclSpace, Kokkos::Cuda>

    Communicator specialization for the :cpp:class:`Experimental::NcclSpace` communication space.
    Wraps an ``ncclComm_t`` handle.

    .. cpp:type:: execution_space = Kokkos::Cuda
    .. cpp:type:: communication_space = Experimental::NcclSpace
    .. cpp:type:: communicator_type = ncclComm_t
    .. cpp:type:: size_type = int
    .. cpp:type:: rank_type = int

    .. cpp:function:: [[nodiscard]] static auto from_raw(ncclComm_t comm, const Kokkos::Cuda& exec = Kokkos::Cuda{}) noexcept -> Communicator<Experimental::NcclSpace, Kokkos::Cuda>

        Constructs a ``Communicator`` from a raw ``ncclComm_t`` handle and a Kokkos CUDA execution space instance. Defaults ``exec`` to ``Kokkos::Cuda``.
        The returned communicator does not own the underlying handle, and the user is responsible for destroying it.

        :param comm: A valid communicator handle.
        :param exec: A Kokkos CUDA execution space instance. Defaults to ``Kokkos::Cuda``.
        :returns: A communicator on success, ``std::nullopt`` if the passed handle was ``nullptr``.

    .. cpp:function:: [[nodiscard]] static auto split_from_raw(const ncclComm_t comm, int color, int key, const Kokkos::Cuda& exec = Kokkos::Cuda{}) noexcept -> std::optional<Communicator<Experimental::NcclSpace, Kokkos::Cuda>>

        Splits from a raw NCCL communicator and associates it to a Kokkos CUDA execution space instance. Defaults ``exec`` to ``Kokkos::Cuda``.

        Creates as many new communicators as distinct values of ``color`` are given, and orders processes according to the value of ``key``. All processes with the same value of ``color`` join the same communicator.
        A process that passes ``NCCL_SPLIT_NOCOLOR`` as ``color`` will not join a new communicator.

        :param comm: A valid communicator handle.
        :param color: A value controlling in which split communicator the calling process should be in.
        :param key: A value ordering the calling process within the split communicator.
        :param exec: A Kokkos CUDA execution space instance. Defaults to ``Kokkos::Cuda``.
        :returns: A split communicator on success, ``std::nullopt`` if the passed color was ``NCCL_SPLIT_NOCOLOR`` or on error.

    .. cpp:function:: [[nodiscard]] static auto duplicate_from_raw(const ncclComm_t comm, const Kokkos::Cuda& exec = Kokkos::Cuda{}) noexcept -> std::optional<Communicator<Experimental::NcclSpace, Kokkos::Cuda>>

        Duplicates from a raw NCCL communicator.

        :param comm: A valid communicator handle.
        :param exec: A Kokkos CUDA execution space instance. Defaults to ``Kokkos::Cuda``.
        :returns: A communicator on success, ``std::nullopt`` on error.

    .. cpp:function:: [[nodiscard]] auto comm() noexcept -> ncclComm_t&
                      [[nodiscard]] auto comm() const noexcept -> const ncclComm_t&

        :returns: A reference to the underlying ``ncclComm_t`` object.

    .. cpp:function:: [[nodiscard]] auto exec() const noexcept -> const Kokkos::Cuda&

        :returns: A const reference to the associated ``Kokkos::Cuda`` execution space instance.


.. _request-ownership:

Requests
--------

.. cpp:class:: template <CommunicationSpace C = DefaultCommunicationSpace> Request

    Template class for request wrappers of different communication space types.

    ``Request`` objects are move-only: copy construction and copy assignment are explicitly deleted.
    Moves transfer the native handle and completion callbacks together, leaving the source null and empty.
    Requests must be explicitly completed before destruction. Destruction never waits, cancels, frees, or completes
    an operation; a non-null handle or pending callbacks trigger a fatal precondition diagnostic.

    Native access through ``request()`` and ``request_ptr()`` is borrowed. Applications must not complete, free,
    replace, or retain an owning copy of the handle: doing so bypasses callback and buffer lifetime management.
    Backend implementations may initialize inactive requests through these accessors.

    :tparam C: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    .. cpp:type:: communication_space = C

    .. cpp:type:: request_type = C::request_type

    .. cpp:type:: rank_type = C::rank_type

Common interfaces
^^^^^^^^^^^^^^^^^

Both specializations share the following interface:

.. cpp:function:: Request(const Request&) = delete

    Copy constructor is deleted because a ``Request`` can only be moved.

.. cpp:function:: auto operator=(const Request&) -> Request& = delete

    Copy assignment operator is deleted because a ``Request`` can only be moved.

.. cpp:function:: Request(Request&& other) noexcept

    Transfers the native handle and callbacks. The source becomes inactive with no pending callbacks.

.. cpp:function:: auto operator=(Request&& other) noexcept -> Request&

    Transfers the native handle and callbacks. The destination must have a null handle and no pending callbacks;
    otherwise a fatal precondition diagnostic is issued. Self-move is a no-op. No operation is implicitly completed.

.. cpp:function:: template <KokkosView V>\
                  auto extend_view_lifetime(const V& view) -> void

    Captures a Kokkos View to extend its lifetime until the request completes.
    Retains allocations owned by managed Views until completion. Has no effect on unmanaged Views: Kokkos Comm
    cannot retain their user-owned allocations, which the application must keep valid until completion.

    :tparam V: A Kokkos View type.
    :param view: The view whose lifetime should be extended.

.. cpp:function:: auto add_callback(std::function<void()>&& cb) -> void

    Implementation hook; applications must not register callbacks directly. Callbacks execute in registration order
    after backend completion, with captured resources retained until all callbacks finish.

    .. warning::

        Callbacks must be nonempty and must not throw exceptions. Kokkos Comm does not use exceptions and assumes
        that callbacks cannot throw; it does not validate callbacks or catch exceptions.
        Callbacks must not mutate, reactivate, destroy, or re-enter
        their owning request, including through ``wait``, ``test``, or batch waits. Violating these requirements is
        unsupported.

    :param cb: The callback function to register.

.. cpp:function:: auto wait() -> void

    Blocks until the associated operation completes and executes all registered callbacks, including unpacking work.
    On return, the native handle is null and callbacks have been released. Calling this on an empty request is harmless.

.. cpp:function:: auto test() -> bool

    Non-blocking query for the completion of the associated operation.
    Executes all registered callbacks if the operation has completed, then leaves a null handle and no callbacks.
    An incomplete operation retains its handle and callbacks. Empty requests return ``true``. Repeated individual
    completion calls do not repeat callbacks.

    :returns: ``true`` if the request has completed, ``false`` otherwise.

.. cpp:function:: auto wait(Request& request) -> void

    Free function overload. Waits on ``request`` until the associated operation completes.

    :param request: A reference to the request to wait on.

.. cpp:function:: auto wait(Request&& request) -> void

    Free function overload. Waits on an r-value ``request``, consuming it upon completion.

    :param request: An r-value reference to the request to wait on.

.. cpp:function:: auto wait_all(std::span<Request> requests) -> void

    Waits for every request that is active on entry. All native handles become null before any completion callbacks
    execute. Then runs and releases every request's callbacks, including callbacks attached to inactive requests.
    An empty span is harmless, and repeated waits do not repeat callbacks.

    :param requests: The list of requests to complete.

.. cpp:function:: auto wait_any(std::span<Request> requests) -> std::optional<rank_type>

    Waits for one active request to complete. The selected request's native handle becomes null, then its callbacks
    execute and are released. Every other request retains its handle and callbacks, even if its operation has already
    completed in the backend. Inactive requests are skipped, including those with pending callbacks; use ``wait`` or
    ``wait_all`` to drain their callbacks. Repeated calls select each active request at most once.

    :param requests: The list of requests to poll.
    :returns: The index within the original span of the selected request, or ``std::nullopt`` if the span is empty or
              all native handles are null. Selection order is unspecified when multiple requests are ready.

.. cpp:function:: auto test(Request& request) -> bool

    Free function overload. Queries ``request`` for completion of the associated operation.

    :param request: A reference to the request to query.
    :returns: ``true`` if the request has completed, ``false`` otherwise.

MPI specialization
^^^^^^^^^^^^^^^^^^

.. cpp:class:: template <> Request<MpiSpace>

    Request specialization for the :cpp:class:`MpiSpace` communication space.
    Wraps an ``MPI_Request`` handle.

    .. cpp:type:: communication_space = MpiSpace
    .. cpp:type:: request_type = MpiSpace::request_type
    .. cpp:type:: rank_type = MpiSpace::rank_type

    .. cpp:function:: Request() noexcept

        Constructs an inactive request with ``MPI_REQUEST_NULL``.

    .. cpp:function:: explicit Request(request_type&& request) noexcept

        Adopts a non-persistent MPI request and sets the source to ``MPI_REQUEST_NULL``. A named raw handle must be
        passed with ``std::move(raw_request)``. Persistent requests are unsupported; use ``Channel<MpiSpace>`` instead.
        Persistence cannot be inspected portably, so non-persistence is a caller precondition.

        Adoption transfers only the request handle. The caller must keep every associated buffer and other operation
        resource valid until completion, including buffers backed by managed Views: adoption supplies no View metadata.

        :param request: The non-persistent ``MPI_Request`` whose ownership is transferred.

    .. cpp:function:: auto request() noexcept -> request_type&
                      auto request() const noexcept -> const request_type&

        :returns: A reference to the underlying ``MPI_Request`` object.

    .. cpp:function:: auto request_ptr() noexcept -> request_type*
                      auto request_ptr() const noexcept -> const request_type*

        :returns: A pointer to the underlying ``MPI_Request`` object.

    .. note::

        Both ``wait_all`` and ``wait_any`` temporarily transfer non-persistent MPI handles into contiguous storage for
        ``MPI_Waitall`` and ``MPI_Waitany``, respectively, incurring an allocation. ``wait_any`` restores non-selected
        handles before callbacks run. MPI statuses are ignored; completion errors are fatal.

NCCL specialization
^^^^^^^^^^^^^^^^^^^

.. cpp:class:: template <> Request<Experimental::NcclSpace>

    Request specialization for the :cpp:class:`Experimental::NcclSpace` communication space.
    Wraps a ``cudaEvent_t`` handle to track the completion of CUDA stream operations.

    .. cpp:type:: communication_space = Experimental::NcclSpace
    .. cpp:type:: request_type = Experimental::NcclSpace::request_type
    .. cpp:type:: rank_type = Experimental::NcclSpace::rank_type

    .. cpp:function:: explicit Request() noexcept

        Constructs an empty ``Request`` with a null event handle.

    .. cpp:function:: ~Request() noexcept

        Requires a null event and no pending callbacks. Violations trigger a fatal diagnostic; no CUDA work is waited
        for and no active event is destroyed.

    .. cpp:function:: auto capture_stream_state(cudaStream_t stream) noexcept -> void

        Implementation hook; applications and completion callbacks must not call it. Records a CUDA event on
        ``stream`` for completion tracking. An already-owned event is a fatal precondition violation; complete the
        request explicitly before reuse. Callbacks registered during operation setup may precede this capture.

        The backend uses one CUDA device per process. The calling thread must have the stream's owning device current.
        Event creation and recording failures are fatal. Successful ``wait`` or ``test`` destroys and nulls the event
        before executing completion callbacks.

        :param stream: The CUDA stream whose state to capture.

    .. cpp:function:: auto request() noexcept -> request_type&
                      auto request() const noexcept -> const request_type&

        :returns: A reference to the underlying ``cudaEvent_t`` object.

    .. cpp:function:: auto request_ptr() noexcept -> request_type*
                      auto request_ptr() const noexcept -> const request_type*

        :returns: A pointer to the underlying ``cudaEvent_t`` object.

    .. note::

        Both ``wait_all`` and ``wait_any`` poll only active events and destroy and null each selected event exactly
        once. ``wait_all`` retires every active event before running callbacks. Active polling increases CPU utilization.


Communication Primitives
========================

Point-to-point
--------------

.. important::

    Use :cpp:func:`exchange` for halo exchanges that require a send and a receive
    to progress together. Separate :cpp:func:`send` and :cpp:func:`recv` calls
    are not a portable substitute: with a blocking NCCL communicator, initiating
    a receive can wait for the peer's send. If every rank initiates its receive
    first, no rank can reach its send, causing a deadlock.

Send
^^^^

.. warning:: This is not a blocking operation despite being named like ``MPI_Send``.

.. cpp:function:: template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto send(Communicator<CommSpace, ExecSpace> &h, SendView &sv, int dest) -> Request<CommSpace>

    Initiates a non-blocking send operation.

    :tparam SendView: The type of the Kokkos view to send.
    :tparam ExecSpace: The execution space to use. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param h: A handle to the execution space and transport mechanism.
    :param sv: The Kokkos view to send.
    :param dest: The destination rank.

    :return: A request object of type ``Request<CommSpace>`` representing the non-blocking send operation.


**Example usage:**

.. literalinclude:: core_send.cpp
    :language: cpp

Receive
^^^^^^^

.. warning:: This is not a blocking operation despite being named like ``MPI_Recv``.

.. cpp:function:: template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto recv(Communicator<CommSpace, ExecSpace> &h, RecvView &sv, int dest) -> Request<CommSpace>

    Initiates a non-blocking receive operation.

    :tparam RecvView: The type of the Kokkos view for receiving data.
    :tparam ExecSpace: The execution space where the operation will be performed. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param h: A handle to the execution space and transport mechanism.
    :param rv: The Kokkos view where the received data will be stored.
    :param src: The source rank from which to receive data.

    :return: A request object of type ``Request<CommSpace>`` representing the non-blocking receive operation.

    This function initiates a non-blocking receive operation using the specified execution space and transport mechanism. The data will be received into the provided view from the specified source rank and message tag. The function returns a request object that can be used to check the status of the receive operation or to wait for its completion.


**Example usage:**

.. literalinclude:: core_recv.cpp
   :language: cpp


Exchange
^^^^^^^^

.. cpp:function:: template <CommunicationSpace Comm, KokkosExecutionSpace Exec, KokkosView SendV, MutKokkosView RecvV> \
                  auto exchange(Communicator<Comm, Exec>& comm, const SendV& sv, int dst, const RecvV& rv, int src) -> Request<Comm>

    Initiates one send to ``dst`` and one receive from ``src`` together, returning
    one request that tracks both operations. The destination and source may be
    different ranks.

    :tparam Comm: The communication backend, deduced from ``comm``.
    :tparam Exec: The execution space, deduced from ``comm``.
    :tparam SendV: The type of the send view. Const-valued views are supported.
    :tparam RecvV: The type of the receive view, whose elements must be mutable.

    :param comm: The communicator and execution space used for both operations.
    :param sv: The view containing the data to send.
    :param dst: The destination rank.
    :param rv: The view where received data will be stored.
    :param src: The source rank.
    :returns: A request representing completion of both the send and the receive.

    Both views may be multi-dimensional and independently contiguous or
    non-contiguous. Non-contiguous send data is packed before communication;
    non-contiguous receive data is unpacked when the request completes.
    The send and receive counts and datatypes need not be the same locally,
    but each message must match its peer's count and datatype. The send and
    receive storage must not overlap.

    The request retains the view handles and any temporary packing buffers.
    Do not modify the send data or access the receive data until completion.
    Calling ``wait()`` waits for both operations and finishes any receive
    unpacking before returning, so the receive view is then ready to use.

    The MPI backend uses ``MPI_Isendrecv`` and requires MPI 4.0 or newer.
    The NCCL backend groups ``ncclSend`` and ``ncclRecv`` so both can progress
    together. Its communicator may block during initiation, but completion of
    the CUDA work is tracked by the returned request.

**Example usage:**

Given initialized communication and Kokkos runtimes, and a valid
``raw_comm_handle`` for the selected backend, all ranks execute this example
to exchange one side of a periodic domain. The column subviews are
non-contiguous; the received left halo is ready after ``wait()``.

.. literalinclude:: core_exchange.cpp
    :language: cpp


Collectives
-----------

.. important::

    Collective operations act **element-wise** on the input Views. Multi-dimensional Views are treated as a **logically flattened** sequence of values, and the reduction is applied over that sequence. All participating Views must have **identical extents**; mismatched shapes result in undefined behavior.

    The reduction operator must be **associative**, but ordering of partial combinations is **not guaranteed**, and the operation is not required to be commutative.

Broadcast
^^^^^^^^^

.. cpp:function:: template <KokkosView View, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto KokkosComm::Experimental::broadcast(Communicator<CommSpace, ExecSpace>& h, View v, int root) -> Request<CommSpace>

    Broadcasts the ``v`` view from the ``root`` rank to all ranks' ``v`` view.

    :tparam View: The type of the Kokkos view to broadcast.
    :tparam ExecSpace: The execution space to use. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param h: A handle to the execution space and transport mechanism.
    :param v: The Kokkos view to broadcast. Must be valid on all ranks; its data on ``root`` is broadcast to all ranks.
    :param root: The rank that holds the source data.

    :returns: A request object of type ``Request<CommSpace>`` representing the non-blocking broadcast operation.

Allgather
^^^^^^^^^

.. cpp:function:: template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto KokkosComm::Experimental::allgather(Communicator<CommSpace, ExecSpace>& h, const SendView sv, RecvView rv) -> Request<CommSpace>

    Gathers the ``sv`` view from each rank into the ``rv`` view on all ranks.

    Data received from rank ``i`` is placed at offset ``i * KokkosComm::span(sv)`` in ``rv``.
    The ``rv`` view must be large enough to hold ``h.size() * KokkosComm::span(sv)`` elements.

    :tparam SendView: The type of the Kokkos view to send from each rank.
    :tparam RecvView: The type of the Kokkos view to receive the gathered data into.
    :tparam ExecSpace: The execution space to use. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param h: A handle to the execution space and transport mechanism.
    :param sv: The Kokkos view to send from this rank.
    :param rv: The Kokkos view where the gathered data from all ranks is stored.

    :returns: A request object of type ``Request<CommSpace>`` representing the non-blocking all-gather operation.

Alltoall
^^^^^^^^

.. cpp:function:: template <KokkosView SendView, KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto KokkosComm::Experimental::alltoall(Communicator<CommSpace, ExecSpace>& h, const SendView sv, RecvView rv, int count) -> Request<CommSpace>

    Sends ``count`` elements to every other rank and receives ``count`` elements from every other rank.

    Data to send to destination rank ``i`` is taken from ``sv`` at offset ``i * count``.
    Data received from source rank ``j`` is placed into ``rv`` at offset ``j * count``.
    Both ``sv`` and ``rv`` must span at least ``h.size() * count`` elements.

    :tparam SendView: The type of the Kokkos view holding the data to send.
    :tparam RecvView: The type of the Kokkos view where the received data will be stored.
    :tparam ExecSpace: The execution space to use. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param h: A handle to the execution space and transport mechanism.
    :param sv: The Kokkos view containing the data to send to all ranks.
    :param rv: The Kokkos view where the received data from all ranks will be stored.
    :param count: The number of elements sent to, and received from, each rank.

    :returns: A request object of type ``Request<CommSpace>`` representing the non-blocking all-to-all operation.

Allreduce
^^^^^^^^^

.. cpp:function:: template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto KokkosComm::Experimental::allreduce(Communicator<CommSpace, ExecSpace>& h, const SendView sv, RecvView rv, RedOp) -> Request<CommSpace>

    Reduces the ``sv`` view using ``RedOp`` and stores the result in all ranks' ``rv`` view.

    :tparam SendView: The type of the Kokkos view to reduce.
    :tparam RecvView: The type of the Kokkos view to store the reduced result into.
    :tparam RedOp: A type satisfying the :cpp:concept:`ReductionOperator` concept.
    :tparam ExecSpace: The execution space to use. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param h: A handle to the execution space and transport mechanism.
    :param sv: The Kokkos view containing the input data to reduce.
    :param rv: The Kokkos view where the result of the reduction is stored on all ranks.

    :returns: A request object of type ``Request<CommSpace>`` representing the non-blocking all-reduce operation.

Reduce
^^^^^^

.. cpp:function:: template <KokkosView SendView, KokkosView RecvView, ReductionOperator RedOp, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto KokkosComm::Experimental::reduce(Communicator<CommSpace, ExecSpace>& h, const SendView sv, RecvView rv, int root, RedOp) -> Request<CommSpace>

    Reduces the ``sv`` view using ``RedOp`` and stores the result in the ``root`` rank's ``rv`` view.

    The ``rv`` view is only used on the ``root`` rank and is ignored on all other ranks.

    :tparam SendView: The type of the Kokkos view to reduce.
    :tparam RecvView: The type of the Kokkos view to store the reduced result into on ``root``.
    :tparam RedOp: A type satisfying the :cpp:concept:`ReductionOperator` concept.
    :tparam ExecSpace: The execution space to use. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param h: A handle to the execution space and transport mechanism.
    :param sv: The Kokkos view containing the input data to reduce.
    :param rv: The Kokkos view where the result of the reduction is stored on ``root``.
    :param root: The rank that receives the reduced result.

    :returns: A request object of type ``Request<CommSpace>`` representing the non-blocking reduce operation.

Utilities
---------

.. cpp:namespace:: KokkosComm

.. warning::

    Non-system data types (i.e. the data types not natively supported by the communication space) are not convertible.
    This notably includes user-defined types.

.. cpp:function:: template <CommunicationSpace C, typename T>\
                  auto datatype() -> C::datatype_type

    Converts a type ``T`` to its communication space ``C`` equivalent representation.

    When ``C`` is:

    * ``MpiSpace``, returns the corresponding ``MPI_Datatype`` type.
    * ``NcclSpace``, returns the corresponding ``ncclDataType_t`` type.

    :tparam C: The target communication space backend to use for data type conversion.
    :tparam T: The C++-native data type to convert from.
    :returns: The communication space representation of the C++-native data type.

.. cpp:function:: template <CommunicationSpace C, KokkosView V>\
                  auto datatype_for([[maybe_unused]] const V& view) -> C::datatype_type

    :tparam C: The target communication space backend to use for data type conversion.
    :tparam V: A Kokkos View type.
    :param view: The Kokkos View to convert the value type from.
    :returns: The communication space representation of the Kokkos View value type.

.. cpp:function:: template <CommunicationSpace C, KokkosView V>\
                  auto datatype_for([[maybe_unused]] C&& comm, [[maybe_unused]] const V& view) -> C::datatype_type

    :tparam C: The target communication space backend to use for data type conversion.
    :tparam V: A Kokkos View type.
    :param comm: A communication space object, immediately consumed.
    :param view: The Kokkos View to convert the value type from.
    :returns: The communication space representation of the Kokkos View value type.

.. cpp:function:: template <CommunicationSpace CS, ReductionOperator RO>\
                  [[nodiscard]] constexpr auto reduction_op() -> CS::reduction_op_type

    Converts a Kokkos Comm :cpp:concept:`ReductionOperator` type to its communication-space-native equivalent.

    When ``CS`` is:

    * ``MpiSpace``, returns the corresponding ``MPI_Op``.
    * ``Experimental::NcclSpace``, returns the corresponding ``ncclRedOp_t``.

    :tparam CS: The target communication space backend.
    :tparam RO: A type satisfying the :cpp:concept:`ReductionOperator` concept.
    :returns: The communication space representation of the reduction operator.

.. cpp:function:: template <CommunicationSpace CS, ReductionOperator RO>\
                  [[nodiscard]] constexpr auto reduction_op_for(RO&& op) -> CS::reduction_op_type

    Instance-based counterpart of :cpp:func:`reduction_op`.

    :tparam CS: The target communication space backend.
    :tparam RO: A type satisfying the :cpp:concept:`ReductionOperator` concept.
    :param op: An instance of a reduction operator tag.
    :returns: The communication space representation of the reduction operator.
