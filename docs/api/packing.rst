Handling Non-continguous Views
==============================

.. cpp:namespace-push:: KokkosComm::Impl::Packing

.. cpp:struct:: template <KokkosView View> \
               MpiArgs

  .. cpp:member:: View view
  .. cpp:member:: MPI_Datatype datatype
  .. cpp:member:: int count

.. cpp:struct:: template <KokkosView View> \
               DeepCopy

    Use Kokkos::deep_copy to translate between non-contiguous and contiguous data.

    .. cpp:type:: args_type = MpiArgs<View>

    .. cpp:type:: non_const_packed_view_type

    .. cpp:function:: template <typename ExecSpace> \
                      static args_type allocate_packed_for(const ExecSpace &space,const std::string &label, const View &src)

        :returns: Return an MpiArgs suitable to hold packed data for ``src``.
        
        Allocates a contiguous Kokkos::View large enough to hold all the data in ``src``.

    .. cpp:function:: template <typename ExecSpace> \
                    static args_type pack(const ExecSpace &space, const View &src)

        Uses allocate_packed_for and Kokkos::deep_copy to return a packed view of the data in ``src``.

    .. cpp:function:: template <typename ExecSpace> \
                    static void unpack_into(const ExecSpace &space, const View &dst, const non_const_packed_view_type &src)

        :param space: The execution space to operate in.
        :param src: A Kokkos::View produced by allocate_packed_for.
        :param dst: A corresponding unpacked Kokkos::View.

        Uses Kokkos::deep_copy to fill ``dst`` with an unpacked view of the data in ``src``.

.. cpp:struct:: template <KokkosView View> \
               MpiDatatype

    Use the MPI Datatype engine to handle non-continguous data

    .. cpp:type:: args_type = MpiArgs<View>

    .. cpp:type:: non_const_packed_view_type

    .. cpp:function:: template <typename ExecSpace> \
                      static args_type allocate_packed_for(const ExecSpace &space,const std::string &label, const View &src)

        :returns: Return an MpiArgs suitable to hold packed data for ``src``.

        Return an MpiArgs holding the ``src``, an MPI_Datatype describing the possibly-non-contiguous data in that Kokkos::View, and a count = 1.

    .. cpp:function:: template <typename ExecSpace> \
                    static args_type pack(const ExecSpace &space, const View &src)

    .. cpp:function:: template <typename ExecSpace> \
                    static void unpack_into(const ExecSpace &space, const View &dst, const non_const_packed_view_type &src)