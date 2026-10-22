****************************
System-Specific Hints & Tips
****************************

El Capitan-like (HIP, Cray MPICH)
=================================

This example is based off of a system like El Capitan @ LLNL

Configure and build Kokkos with HIP support enabled:

.. code-block:: bash

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

.. code-block:: bash

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


El Capitan-like (HIP, Cray MPICH, RCCL)
=======================================

This example is based off of a system like El Capitan @ LLNL

.. code-block:: bash

    flux alloc -N 1 -t 4h -q short

    module load cmake/3.29.2 cce/21.0.2 rocmcc/7.2.1-cce-21.0.2-magic cray-mpich rccl/working-env

    export CRAYPE_LINK_TYPE=dynamic

    cmake -S kokkos -B build-kokkos \
            -DCMAKE_CXX_COMPILER=hipcc \
            -DKokkos_ENABLE_HIP=ON \
            -DKokkos_ARCH_AMD_GFX942_APU=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -Dhip_DIR=/opt/rocm-7.2.1/lib/cmake/hip/ \
            -Drocprim_DIR=/opt/rocm-7.2.1/lib/cmake/rocprim/

    cmake --build build-kokkos -j $(nproc)
    cmake --install build-kokkos --prefix install-kokkos

Configure Kokkos Comm with MPI + RCCL enabled

.. code-block:: bash

    # need to reset the language on the link line so compiler doesn't treat librares as hip
    cmake -S . -B build-kc \
        -DCMAKE_C_COMPILER=mpiamdclang \
        -DCMAKE_CXX_COMPILER=mpiamdclang++ \
        -DKokkos_ROOT=$(realpath install-kokkos) \
        -DKokkosComm_ENABLE_MPI=ON \
        -DKokkosComm_ENABLE_RCCL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DKokkosComm_ENABLE_TESTS=ON \
        -Drocprim_DIR=/opt/rocm-7.2.1/lib/cmake/rocprim/ \
        -Drccl_DIR=/opt/rocm-7.2.1/lib/cmake/rccl/ \
        -DCMAKE_CXX_FLAGS="-D__HIP_PLATFORM_AMD__ -I/opt/rocm-7.2.1/include --offload-arch=gfx942 --rocm-path=/opt/rocm-7.2.1 -x hip" \
        -DCMAKE_EXE_LINKER_FLAGS="-x none -L/opt/rocm-7.2.1/lib -lrccl" \
        -DMPIEXEC_EXECUTABLE:FILEPATH="$(command -v flux)" \
        -DMPIEXEC_NUMPROC_FLAG='run;-N1;-n' \
        -DMPIEXEC_PREFLAGS='--exclusive'

    cmake --build build-kc -j $(nproc) --target test.core.send-recv

    export HSA_XNACK=1
    export MPICH_GPU_SUPPORT_ENABLED=1
    export MPICH_SMP_SINGLE_COPY_MODE=XPMEM
    ctest --test-dir build-kc -V
