*****************************
Handling non-contiguous views
*****************************

MPI
===

.. cpp:namespace-push:: KokkosComm::mpi::Impl::Packer

.. cpp:struct:: template <KokkosView View> MpiArgs

    A wrapper type describing a Kokkos view for the MPI communication backend.

    .. cpp:member:: View view

        A reference to the Kokkos view to communicate.

    .. cpp:member:: MPI_Datatype datatype

        The corresponding MPI data type of the Kokkos view.

    .. cpp:member:: int count

        The number of elements in the Kokkos view.


.. cpp:struct:: template <KokkosView View> DeepCopy

    Use ``Kokkos::deep_copy`` to translate between non-contiguous and contiguous data.

    .. cpp:type:: non_const_packed_view_type = Kokkos::View<typename View::non_const_data_type, Kokkos::LayoutLeft, typename View::memory_space>

    .. cpp:type:: args_type = MpiArgs<non_const_packed_view_type>


    .. cpp:function:: template <KokkosExecutionSpace ExecSpace> \
                      static auto allocate_packed_for(const ExecSpace &space, const std::string &label, const View &src)-> args_type

        Allocates  contiguous Kokkos view large enough to hold all the data in ``src``.

        :param space: The execution space to operate in.
        :param label: Identification label for the allocation.
        :param src: A Kokkos::View produced by ``allocate_packed_for``.

        :returns: Return an ``MpiArgs`` object suitable to hold packed data for ``src``.


    .. cpp:function:: template <KokkosExecutionSpace ExecSpace> \
                      static auto pack(const ExecSpace &space, const View &src) -> args_type

        Uses ``allocate_packed_for`` and ``Kokkos::deep_copy`` to pack the data in ``src``.

        :param space: The execution space to operate in.
        :param src: A Kokkos view produced by ``allocate_packed_for``.

        :return: A packed Kokkos view of the data in ``src``.


    .. cpp:function:: template <KokkosExecutionSpace ExecSpace> \
                      static auto unpack_into(const ExecSpace &space, const View &dst, const non_const_packed_view_type &src) -> void

        Uses ``Kokkos::deep_copy`` to fill ``dst`` with an unpacked view of the data in ``src``.

        :param space: The execution space to operate in.
        :param src: A Kokkos view produced by ``allocate_packed_for``.
        :param dst: A corresponding unpacked Kokkos view.


.. cpp:struct:: template <KokkosView View> MpiDatatype

    Use the MPI data type engine to handle non-contiguous data.

    .. cpp:type:: non_const_packed_view_type = View

    .. cpp:type:: args_type = MpiArgs<non_const_packed_view_type>


    .. cpp:function:: template <KokkosExecutionSpace ExecSpace> \
                      static auto allocate_packed_for(const ExecSpace &space, const std::string &label, const View &src) -> args_type

        Does not actually allocate. Constructs an ``MpiArgs`` describing the view and returns it:

            * the ``src``;
            * an ``MPI_Datatype`` describing the possibly-non-contiguous data in that Kokkos::View, and;
            * a count equal to 1.

        :param space: The execution space to operate in.
        :param label: Identification label for the allocation.
        :param src: A Kokkos view produced by ``allocate_packed_for``.

        :returns: Return an ``MpiArgs`` suitable to hold packed data for ``src``.

    .. cpp:function:: template <KokkosExecutionSpace ExecSpace> \
                      static auto pack(const ExecSpace &space, const View &src) -> args_type

        No-op, rely on MPI's data type engine.

        :param space: The execution space to operate in.
        :param src: A Kokkos view produced by ``allocate_packed_for``.

        :return: A packed Kokkos view of the data in ``src``.

    .. cpp:function:: template <KokkosExecutionSpace ExecSpace> \
                      static auto unpack_into(const ExecSpace &space, const View &dst, const non_const_packed_view_type &src) -> void

        No-op, rely on MPI's data type engine.

        :param space: The execution space to operate in.
        :param src: A Kokkos view produced by ``allocate_packed_for``.
        :param dst: A corresponding unpacked Kokkos view.
