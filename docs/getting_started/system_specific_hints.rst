****************************
System-Specific Hints & Tips
****************************

AMD GPU systems with Cray MPICH
==================================

This example is based off of a system like El Capitan @ LLNL

Configure and build Kokkos with HIP support enabled:

.. code-block:: console

    module load cce/20.0.0-magic rocmcc cray-mpich
    export CRAYPE_LINK_TYPE=dynamic
    export HSA_XNACK=1

    cmake -S kokkos -B build-kokkos \
            -DCMAKE_CXX_COMPILER=mpicxx \
            -DKokkos_ENABLE_HIP=ON \
            -DKokkos_ARCH_AMD_GFX942_APU=ON \
            -DCMAKE_BUILD_TYPE=Release

    cmake --build build-kokkos -j $(nproc)
    cmake --install build-kokkos --prefix install-kokkos

.. note:: Choose a GPU architecture appropriate for the system


Configure Kokkos Comm with MPI enabled:

.. code-block:: console

    cmake -S . -B build-kc \
            -DCMAKE_C_COMPILER=mpicc \
            -DCMAKE_CXX_COMPILER=mpicxx \
            -DKokkos_ROOT=$(realpath install-kokkos) \
            -DKokkosComm_ENABLE_MPI=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DKokkosComm_ENABLE_TESTS=ON \
            -DCMAKE_EXE_LINKER_FLAGS="-lmpi_gtl_hsa -lxpmem"

    cmake --build build-kc -j $(nproc)

    export MPICH_GPU_SUPPORT_ENABLED=1
    ctest -V --test-dir build-kc
