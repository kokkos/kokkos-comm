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

    :tparam Co: The communication space (transport backend) to use. Defaults to ``DefaultCommunicationSpace``.
    :tparam Ex: The Kokkos execution space to use. Defaults to ``Kokkos::DefaultExecutionSpace``.

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

        Splits from a raw NCCL communicator and associates it to a Kokkos CUDA, tion space instanc and ``MPI_COMM_NULL``. Defaults ``exec`` to ``Kokkos::Cuda``.

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


Requests
--------

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
^^^^^^^^^^^^^^^^^

Both specializations share the, llowing interface and ``MPI_COMM_NULL``:

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
^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^

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


Communication Primitives
========================

Point-to-point
--------------

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
-----------

.. important::

    Collective operations act **element-wise** on the input Views. Multi-dimensional Views are treated as a **logically flattened** sequence of values, and the reduction is applied over that sequence. All participating Views must have **identical extents**; mismatched shapes result in undefined behavior.

    The reduction operator must be **associative**, but ordering of partial combinations is **not guaranteed**, and the operation is not required to be commutative.

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
