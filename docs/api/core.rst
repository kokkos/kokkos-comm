****
Core
****

Point-to-point
==============

.. cpp:namespace:: KokkosComm

Send
----

.. warning:: This is not a blocking operation despite being named like ``MPI_Send``.

.. cpp:function:: template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto send(Handle<ExecSpace, CommSpace> &h, SendView &sv, int dest) -> Request<CommSpace>

    Initiates a non-blocking send operation.

    :tparam SendView: The type of the Kokkos view to send.
    :tparam ExecSpace: The execution space to use. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param h: A handle to the execution space and transport mechanism.
    :param sv: The Kokkos view to send.
    :param dest: The destination rank.

    :return: A request object of type ``Request<CommSpace>`` representing the non-blocking send operation.


.. cpp:function:: template <KokkosView SendView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto send(SendView &sv, int dest) -> Request<CommSpace>

    Initiates a non-blocking send operation using a default handle.

    :tparam SendView: The type of the Kokkos view to send.
    :tparam ExecSpace: The execution space to use. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param sv: The Kokkos view to send.
    :param dest: The destination rank.

    :return: A request object of type ``Request<CommSpace>`` representing the non-blocking send operation.

**Example usage:**

.. literalinclude:: core_send.cpp
    :language: cpp

Receive
-------

.. warning:: This is not a blocking operation despite being named like ``MPI_Recv``.

.. cpp:function:: template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto recv(Handle<ExecSpace, CommSpace> &h, RecvView &sv, int dest) -> Request<CommSpace>

    Initiates a non-blocking receive operation.

    :tparam RecvView: The type of the Kokkos view for receiving data.
    :tparam ExecSpace: The execution space where the operation will be performed. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param h: A handle to the execution space and transport mechanism.
    :param rv: The Kokkos view where the received data will be stored.
    :param src: The source rank from which to receive data.

    :return: A request object of type ``Request<CommSpace>`` representing the non-blocking receive operation.

    This function initiates a non-blocking receive operation using the specified execution space and transport mechanism. The data will be received into the provided view from the specified source rank and message tag. The function returns a request object that can be used to check the status of the receive operation or to wait for its completion.


.. cpp:function:: template <KokkosView RecvView, KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto recv(RecvView &sv, int dest) -> Request<CommSpace>

    Initiates a non-blocking receive operation using a default handle.

    :tparam RecvView: The type of the Kokkos view for receiving data.
    :tparam ExecSpace: The execution space where the operation will be performed. Defaults to `Kokkos::DefaultExecutionSpace`.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param rv: The Kokkos view where the received data will be stored.
    :param src: The source rank from which to receive data.

    :return: A request object of type ``Request<CommSpace>`` representing the non-blocking receive operation.

**Example usage:**

.. literalinclude:: core_recv.cpp
   :language: cpp


Collectives
===========

.. important::

    Collective operations act **element-wise** on the input Views. Multi-dimensional Views are treated as a **logically flattened** sequence of values, and the reduction is applied over that sequence. All participating Views must have **identical extents**; mismatched shapes result in undefined behavior.

    The reduction operator must be **associative**, but ordering of partial combinations is **not guaranteed**, and the operation is not required to be commutative.

.. cpp:namespace:: KokkosComm

.. cpp:function:: template <KokkosExecutionSpace ExecSpace = Kokkos::DefaultExecutionSpace, CommunicationSpace CommSpace = DefaultCommunicationSpace> auto barrier(Handle<ExecSpace, CommSpace> &&h) -> void

    A function to create a barrier using the given execution space and transport handle.

    :tparam ExecSpace: The execution space where the operation will be performed. Defaults to ``Kokkos::DefaultExecutionSpace``.
    :tparam CommSpace: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    :param h: A handle of type ``Handle<ExecSpace, CommSpace>`` to be forwarded to the barrier implementation.


Related types
=============

.. cpp:namespace:: KokkosComm

.. cpp:class:: template <CommunicationSpace C = DefaultCommSpace> Request

    Template class for request wrappers of different communication space types.

    ``Request`` objects are move-only: copy construction and copy assignment are explicitly deleted.
    There is always exactly one owner of a ``Request`` and its associated callbacks. This design ensures it is
    impossible for the same callback to be executed more than once.

    :tparam C: The communication backend to use. Defaults to ``DefaultCommunicationSpace``.

    .. cpp:type:: communication_space = C

    .. cpp:type:: request_type = C::request_type

    .. cpp:type:: rank_type = C::rank_type

Common interfaces
-----------------

Both specializations share the following interface:

.. cpp:function:: Request(const Request&) = delete

    Copy constructor is deleted because a ``Request`` can only be moved.

.. cpp:function:: auto operator=(const Request&) -> Request& = delete

    Copy assignment operator is deleted because a ``Request`` can only be moved.

.. cpp:function:: Request(Request&&) = default

    Move constructor is defaulted.

.. cpp:function:: auto operator=(Request&&) -> Request& = default

    Move assignment operator is defaulted.

.. cpp:function:: template <KokkosView V>\
                  auto extend_view_lifetime(const V& view) -> void

    Captures a Kokkos View to extend its lifetime until the request completes.
    Has no effect on unmanaged Views.

    :tparam V: A Kokkos View type.
    :param view: The view whose lifetime should be extended.

.. cpp:function:: auto add_callback(std::function<void()>&& cb) -> void

    Registers a callback to be invoked after the request completes.

    :param cb: The callback function to register.

.. cpp:function:: auto wait() -> void

    Blocks until the associated operation completes. Executes all registered callbacks upon completion.

.. cpp:function:: auto test() -> bool

    Non-blocking query for the completion of the associated operation.
    Executes all registered callbacks if the operation has completed.

    :returns: ``true`` if the request has completed, ``false`` otherwise.

.. cpp:function:: auto wait(Request& request) -> void

    Free function overload. Waits on ``request`` until the associated operation completes.

    :param request: A reference to the request to wait on.

.. cpp:function:: auto wait(Request&& request) -> void

    Free function overload. Waits on an r-value ``request``, consuming it upon completion.

    :param request: An r-value reference to the request to wait on.

.. cpp:function:: auto wait_all(std::span<Request> requests) -> void

    Waits for the completion of all requests in ``requests``.

    :param requests: The list of requests to complete.

.. cpp:function:: auto wait_any(std::span<Request> requests) -> std::optional<rank_type>

    Waits for the completion of at least one request in ``requests``.

    :param requests: The list of requests to poll.
    :returns: The index of the completed request, or ``std::nullopt`` if ``requests`` is empty.

.. cpp:function:: auto test(Request& request) -> bool

    Free function overload. Queries ``request`` for completion of the associated operation.

    :param request: A reference to the request to query.
    :returns: ``true`` if the request has completed, ``false`` otherwise.

MPI specialization
------------------

.. cpp:class:: template <> Request<MpiSpace>

    Request specialization for the :cpp:class:`MpiSpace` communication space.
    Wraps an ``MPI_Request`` handle.

    .. cpp:type:: communication_space = MpiSpace
    .. cpp:type:: request_type = MpiSpace::request_type
    .. cpp:type:: rank_type = MpiSpace::rank_type

    .. cpp:function:: explicit Request(request_type request = MPI_REQUEST_NULL)

        Constructs a ``Request`` from an ``MPI_Request`` handle.

        :param request: The ``MPI_Request`` to encapsulate. Defaults to ``MPI_REQUEST_NULL``.

    .. cpp:function:: auto request() noexcept -> request_type&
                      auto request() const noexcept -> const request_type&

        :returns: A reference to the underlying ``MPI_Request`` object.

    .. cpp:function:: auto request_ptr() noexcept -> request_type*
                      auto request_ptr() const noexcept -> const request_type*

        :returns: A pointer to the underlying ``MPI_Request`` object.

    .. note::

        Both ``wait_all`` and ``wait_any`` copy the underlying ``MPI_Request`` objects into an intermediate container
        before calling ``MPI_Waitall`` and ``MPI_Waitany``, respectively, which incurs an allocation overhead.

NCCL specialization
-------------------

.. cpp:class:: template <> Request<Experimental::NcclSpace>

    Request specialization for the :cpp:class:`Experimental::NcclSpace` communication space.
    Wraps a ``cudaEvent_t`` handle to track the completion of CUDA stream operations.

    .. cpp:type:: communication_space = Experimental::NcclSpace
    .. cpp:type:: request_type = Experimental::NcclSpace::request_type
    .. cpp:type:: rank_type = Experimental::NcclSpace::rank_type

    .. cpp:function:: explicit Request()

        Constructs an empty ``Request`` with a null event handle.

    .. cpp:function:: ~Request() noexcept

        Destructor. Destroys the underlying ``cudaEvent_t`` if one has been created.

    .. cpp:function:: auto capture_stream_state(cudaStream_t stream) noexcept -> void

        Records a CUDA event on ``stream`` to capture its current state for completion tracking.
        If a ``cudaEvent_t`` was previously created on this request, it is destroyed first.

        :param stream: The CUDA stream whose state to capture.

    .. cpp:function:: auto request() noexcept -> request_type&
                      auto request() const noexcept -> const request_type&

        :returns: A reference to the underlying ``cudaEvent_t`` object.

    .. cpp:function:: auto request_ptr() noexcept -> request_type*
                      auto request_ptr() const noexcept -> const request_type*

        :returns: A pointer to the underlying ``cudaEvent_t`` object.

    .. note::

        Both ``wait_all`` and ``wait_any`` use active polling loops rather than blocking synchronization. While this
        increases CPU utilization, it avoids the overhead of spawning threads or completing requests sequentially.
