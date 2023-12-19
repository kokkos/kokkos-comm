Point-to-point
==============

.. cpp:function:: template <typename SendView, typename ExecSpace> \
                  void Kokkos::send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm)

    MPI_Send wrapper.

    :param space: The execution space to operate in
    :param sv: The data to send
    :param dest: the destination rank
    :param tag: the MPI tag
    :param comm: the MPI communicator
    :tparam SendView: A Kokkos::View to send
    :tparam ExecSpace: A Kokkos execution space to operate in