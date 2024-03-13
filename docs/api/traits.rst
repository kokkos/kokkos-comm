Traits
======

Basic Traits
------------

.. cpp:type:: template<typename View> \
              KokkosComm::Traits

    A common interface to access Kokkos::View- and std::mdspan-like types.

    .. cpp:function:: static bool is_contiguous(const View &v)

        :param v: The View to query
        :returns: true iff the data in the ``v`` is contiguous.

    .. cpp:function:: static auto data_handle(const View &v)

    .. cpp:function:: static size_t span(const View &v)

        :returns: the number of bytes between the beginning of the first byte and the end of the last byte of data in ``v``.

        For example, if ``View`` was an std::vector<int16_t> of size 3, it would be 6.
        If the ``View`` is non-contiguous, the result includes any "holes" in ``v``.

    .. cpp:function:: static constexpr bool is_reference_counted()

        :returns: true iff the type is subject to reference counting (e.g., Kokkos::View)

        This is used to determine if asynchronous MPI operations may need to extend the lifetime of this type when it's used as an argument.

    .. cpp:function:: static constexpr size_t rank()

        :returns: the rank (number of dimensions) of the ``View`` type

Packing Traits
--------------

Strategies for handling non-contiguous views

.. cpp:type:: template<typename View> \
              KokkosComm::PackTraits

    A common packing-related interface for Kokkos::View- and std::mdspan-like types.

  .. cpp:type:: packer_type

    The Packer to use for this ``View`` type.

  .. cpp:function:: static bool needs_unpack(const View &v)

    :returns: true iff ``View`` ``v`` needs to be packed before being passed to MPI

  .. cpp:function:: static bool needs_pack(const View &v)

    :returns: true iff ``View`` ``v`` needs to be unpacked after being passed from MPI