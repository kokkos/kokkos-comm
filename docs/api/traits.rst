******
Traits
******

General traits
--------------

.. cpp:namespace:: KokkosComm

.. cpp:struct:: template <KokkosView V> Traits<V>

    A struct that can be specialized to implement custom behavior for a particular Kokkos View.

    .. cpp:type:: non_const_packed_view_type = Kokkos::View<typename V::non_const_data_type, typename V::execution_space::array_layout, typename V::memory_space>

    .. cpp:type:: packed_view_type = Kokkos::View<typename V::data_type, typename V::execution_space::array_layout, typename V::memory_space>


.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] constexpr auto data_handle(const V& view) noexcept -> V::pointer_type

    :tparam V: A Kokkos View type.

    :param view: The Kokkos View to query.

    :returns: A pointer to the underlying data allocation.


.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] constexpr auto rank() noexcept -> size_t

.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] constexpr auto rank([[maybe_unused]] const V& view) noexcept -> size_t

    :tparam V: A Kokkos view type.

    :param view: The Kokkos view to query.

    :returns: The rank (number of dimensions) of the View.


.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] constexpr auto size(const V& view) noexcept -> size_t

    :tparam V: A Kokkos View type.

    :param view: The Kokkos View to query.

    :returns: The product of extents, i.e., the logical number of elements in the View.


.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] constexpr auto span(const V& view) noexcept -> V::size_type

    :tparam V: A Kokkos View type.

    :param view: The Kokkos View to query.

    :returns: The span between the elements of lowest and highest address.

    The span may be larger than the product of extents due to padding, and or non-contiguous data layout.


.. cpp:function:: template <KokkosView V, std::integral I> \
                  [[nodiscard]] constexpr auto extent(const V& view, I i) noexcept -> size_t

    :tparam V: A Kokkos view type.
    :tparam I: An integral type.

    :param view: The Kokkos view to query.
    :param i: The index of the dimension. Must be smaller than the rank of the View.

    :returns: The extent (number of elements) of the specified dimension.


.. cpp:function:: template <KokkosView V, std::integral I> \
                  [[nodiscard]] constexpr auto stride(const V& view, I i) noexcept -> V::size_type

    :tparam V: A Kokkos view type.
    :tparam I: An integral type.

    :param view: The Kokkos view to query.
    :param i: The index of the dimension. Must be smaller than the rank of the View.

    :returns: The stride (number of elements the mapping advances upon increment) of the specified dimension.


.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] constexpr auto is_reference_counted() noexcept -> bool
.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] constexpr auto is_reference_counted([[maybe_unused]] const V& view) noexcept -> bool

    :tparam V: A Kokkos view type.

    :param view: The Kokkos view to query.

    :returns: True if, and only if, the type is subject to reference counting (e.g., always true for ``Kokkos::View`` objects).

    This is used to determine if asynchronous communication operations may need to extend the lifetime of this type when it is used as an argument.


.. cpp:function:: template <KokkosView V> \
                  [[nodiscard]] auto is_contiguous(const V& view) noexcept -> bool

    Checks if a view is contiguous in memory.

    :tparam V: A Kokkos view type.

    :param view: The Kokkos view to query.

    :returns: True if, and only if, the product of extents is equal to the span.


Packing Traits
--------------

Strategies for handling non-contiguous views.

.. cpp:namespace:: KokkosComm::mpi::Impl

.. cpp:struct:: template<typename T> PackTraits<T>

    A common packing-related struct that can be specialized to implement custom behavior for a particular Kokkos view.

    .. cpp:type:: packer_type = Packer::DeepCopy<View>

    The packer to use for this ``View`` type.

.. .. cpp:function:: static auto needs_unpack(const View &v) -> bool

..     :returns: True if, and only if, the ``v`` needs to be unpacked before being passed to the communication backend.

.. .. cpp:function:: static auto needs_pack(const View &v) -> bool

..     :returns: True if, and only if, the ``v`` needs to be packed before being passed to the communication backend.
