Core
====

.. list-table:: MPI API Support
    :widths: 40 30 15 15
    :header-rows: 1

    * - MPI
      - ``KokkosComm::``
      - ``Kokkos::View``
    * - MPI_Send
      - send
      - ✓
      - ✓
    * - MPI_Recv
      - recv
      - ✓
      - ✓
    * - MPI_Isend
      - isend
      - ✓
      - ✓
    * - MPI_Reduce
      - reduce
      - ✓
      - ✓

Point-to-point
--------------

.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView> \
                  Req KokkosComm::isend(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm)

    MPI_Isend wrapper

    :param space: The execution space to operate in
    :param sv: The data to send
    :param dest: the destination rank
    :param tag: the MPI tag
    :param comm: the MPI communicator
    :tparam SendView: A Kokkos::View to send
    :tparam ExecSpace: A Kokkos execution space to operate in
    :returns: A KokkosComm::Req representing the asynchronous communication and any lifetime-extended views.

.. cpp:function:: template <KokkosExecutionSpace ExecSpace, KokkosView SendView> \
                  void KokkosComm::send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm)

    MPI_Send wrapper

    :param space: The execution space to operate in
    :param sv: The data to send
    :param dest: the destination rank
    :param tag: the MPI tag
    :param comm: the MPI communicator
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
