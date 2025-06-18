******************************
As a Trilinos external package
******************************

Kokkos Comm can be used as an external package in Trilinos as of https://github.com/trilinos/Trilinos/pull/14135.

.. code-block:: console

  # acquire trilinos
  git clone https://github.com/trilinos/Trilinos.git

  # symlink kokkos-comm source into Trilinos
  ln -s kokkos-comm trilinos/packages/kokkos-comm

  # Build Trilinos
  cmake -S trilinos -B "$TRILINOS_BUILD" \
    -DCMAKE_INSTALL_PREFIX="$TRILINOS_INSTALL" \
    -DTPL_ENABLE_MPI=ON \
    -DTrilinos_ENABLE_KokkosComm=ON \
    -DTrilinos_ENABLE_Tpetra=ON \
      -DTpetra_ENABLE_TESTS=ON

Look for something like the following in the Trilinos configure output:

.. code-block:: text

  ...
  Final set of enabled top-level packages:  Kokkos Teuchos KokkosKernels KokkosComm Tpetra 5
  ...
  Final set of enabled external packages/TPLs:  MPI BLAS LAPACK DLlib 4
  ...
