*******************
Concepts and Traits
*******************

Concepts
========

.. cpp:namespace:: KokkosComm

.. cpp:concept:: template <typename T> KokkosView

    Specifies that a type ``T`` is a ``Kokkos::View`` object.


.. cpp:concept:: template <typename T> KokkosExecutionSpace

    Specifies that a type ``T`` is a ``Kokkos::ExecutionSpace``.


.. cpp:concept:: template <typename T> CommunicationSpace

    Specifies that a type ``T`` is a KokkosComm communication backend.


******
Traits
******

General traits
--------------

.. cpp:namespace:: KokkosComm

.. cpp:struct:: template <KokkosView V> Traits<V>

    A struct that can be specialized to implement custom behavior for a particular Kokkos view.

    .. cpp:type:: non_const_packed_view_type = Kokkos::View<typename V::non_const_data_type, typename V::execution_space::array_layout, typename V::memory_space>

    .. cpp:type:: packed_view_type = Kokkos::View<typename V::data_type, typename V::execution_space::array_layout, typename V::memory_space>


.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] constexpr auto data_handle(const V& view) -> V::pointer_type

    :tparam V: The type of the Kokkos view.

    :param view: The Kokkos view to query.

    :returns: The pointer to the underlying data allocation.


.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] constexpr auto span(const V& view) -> V::size_type

    :tparam V: The type of the Kokkos view.

    :param view: The Kokkos view to query.

    :returns: The number of bytes between the beginning of the first byte and the end of the last byte of data in ``view``.

    For example, if ``V`` is a ``Kokkos::View<int16_t[3]>``, its span would be 6 (3 elements times 2 bytes).
    If the view is non-contiguous, the result includes any "holes" in ``view``.


.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] constexpr auto rank() -> V::size_type
.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] constexpr auto rank([[maybe_unused]] const V& view) -> V::size_type

    :tparam V: The type of the Kokkos view.

    :param v: The Kokkos view to query.

    :returns: The rank (number of dimensions) of the view type ``V``.


.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] constexpr auto extent(const V& view, int i) -> V::size_type

    :tparam View: The type of the Kokkos view.

    :param v: The Kokkos view to query.
    :param i: The index of the dimension. Must be smaller than the ``rank`` of the view.

    :returns: The extent of the specified dimension.


.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] constexpr auto stride(const V& view, int i) -> V::size_type

    :tparam View: The type of the Kokkos view.

    :param v: The Kokkos view to query.
    :param i: The index of the dimension. Must be smaller than the ``rank`` of the view.

    :returns: The stride of the specified dimension.


.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] constexpr auto is_reference_counted() -> bool

    :tparam View: The type of the Kokkos view.

    :returns: True if, and only if, the type is subject to reference counting (e.g., always true for ``Kokkos::View`` objects).

    This is used to determine if asynchronous MPI operations may need to extend the lifetime of this type when it's used as an argument.


.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] auto is_contiguous(const V& view) -> bool

    Checks if a view is contiguous in memory.

    :tparam View: The type of the Kokkos view.

    :param v: The Kokkos view to query.

    :returns: True if, and only if, the product of extents is equal to the span (i.e., the data in ``view`` is contiguous).


Packing Traits
--------------

Strategies for handling non-contiguous views.

.. cpp:namespace:: KokkosComm

.. cpp:struct:: template<typename T> PackTraits<T>

    A common packing-related struct that can be specialized to implement custom behavior for a particular Kokkos view.

    .. cpp:type:: packer_type = Impl::Packer::DeepCopy<View>

    The packer to use for this ``View`` type.

.. .. cpp:function:: static auto needs_unpack(const View &v) -> bool

..     :returns: True if, and only if, the ``v`` needs to be unpacked before being passed to the communication backend.

.. .. cpp:function:: static auto needs_pack(const View &v) -> bool

..     :returns: True if, and only if, the ``v`` needs to be packed before being passed to the communication backend.
