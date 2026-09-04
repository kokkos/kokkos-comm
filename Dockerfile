ARG CUDA_VERSION=12.9.0
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04
ARG MPI_VENDOR=openmpi
FROM ${BASE_IMAGE} AS base
LABEL authors="cedricchevalier19"
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C
ARG KOKKOS_VERSION=5.2.0

# Install dependencies and kitware repository for latest cmake
RUN --mount=type=cache,target=/var/lib/apt/lists <<'EOF' bash
    set -exuo pipefail
    DISTRIB_CODENAME=""
    [ -r /etc/lsb-release ] && . /etc/lsb-release
    if [ -n "${DISTRIB_CODENAME}" ]; then
        apt-get update && apt-get install -y --no-install-recommends apt-transport-https gnupg software-properties-common curl
        curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc  2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
        echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $DISTRIB_CODENAME main" > /etc/apt/sources.list.d/kitware.list
        apt-get update && apt-get install -y --no-install-recommends kitware-archive-keyring;
    fi
EOF

RUN --mount=type=cache,target=/var/lib/apt/lists <<EOF bash
    set -exuo pipefail
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache cmake curl gdb git jq libnuma-dev less ninja-build
EOF

ENV CUDA_ROOT=/usr/local/cuda
ENV PATH=${CUDA_ROOT}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${PATH}
ENV NVCC_CCBIN=/usr/bin/gcc

FROM base AS kokkos
ENV Kokkos_ROOT=/usr/local/kokkos
ARG CUDA_ARCH=HOPPER90
# Download and install Kokkos \
RUN <<'EOF' bash
    set -exuo pipefail
    URL="$(curl -s "https://api.github.com/repos/kokkos/kokkos/releases/tags/${KOKKOS_VERSION}" | jq -r '.tarball_url')"
    mkdir kokkos && curl -fsSL "$URL" | tar -xzf - -C kokkos --strip-components=1
    cd kokkos
    CMAKE_FLAGS=$([ -r "${CUDA_ROOT}" ] && echo "-DKokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_${CUDA_ARCH}=ON" || echo "")
    cmake -B build -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=${Kokkos_ROOT} \
     ${CMAKE_FLAGS} -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=ON
    cmake --build build -j$(nproc)
    cmake --install build
    cd ..
    rm -rf kokkos
EOF

FROM base AS mpi
ENV MPI_ROOT=/usr/local/mpi
ENV PATH=${MPI_ROOT}/bin:${PATH}
ENV LD_LIBRARY_PATH=${MPI_ROOT}/lib64:${PATH}

FROM mpi as openmpi
ARG OPENMPI_URL=https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.8.tar.gz
# Compile and install openmpi from source, with CUDA support
RUN --mount=type=cache,target=/root/.cache/ccache <<'EOF' bash
    set -exuo pipefail
    mkdir mpi && curl -fsSL "${OPENMPI_URL}" | tar -xzf - -C mpi --strip-components=1
    cd mpi
    CONFIGURE_FLAGS=$([ -r "${CUDA_ROOT}" ] && echo "--with-cuda=${CUDA_ROOT}" || echo "")
    CC="ccache gcc" CXX="ccache g++" NVCC="ccache nvcc" ./configure ${CONFIGURE_FLAGS} --prefix=${MPI_ROOT}
    make -j$(nproc)
    make install
    ldconfig
    cd ..
    rm -rf mpi
    ccache -s
# Allow to run mpi program as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
EOF

FROM mpi AS mpich
ARG MPICH_URL=https://www.mpich.org/static/downloads/5.0.0/mpich-5.0.0.tar.gz
# Filter out `ccbin` option that conflicts with ccache. Rely on NVCC_BIN.
ADD containers/mpich-nvcc.patch .
# Compile and install openmpi from source, with CUDA support
RUN --mount=type=cache,target=/root/.cache/ccache <<'EOF' bash
    set -exuo pipefail
    mkdir mpi && curl -fsSL "${MPICH_URL}" | tar -xzf - -C mpi --strip-components=1
    cd mpi
    patch -p1 < ../mpich-nvcc.patch
    CONFIGURE_FLAGS=$([ -r "${CUDA_ROOT}" ] && echo "--with-cuda=${CUDA_ROOT}" || echo "")
    CC="ccache gcc" CXX="ccache g++" NVCC="ccache nvcc" NVCC_BIN=/usr/bin/gcc ./configure ${CONFIGURE_FLAGS} --prefix=${MPI_ROOT} --disable-fortran
    make -j$(nproc)
    make install
    ldconfig
    cd ..
    rm -rf mpi
    ccache -s
EOF

FROM ${MPI_VENDOR} AS workhorse
COPY --from=kokkos /usr/local/kokkos/ /usr/local/kokkos/

FROM workhorse AS devhorse
ADD docs/requirements.txt .
RUN --mount=type=cache,target=/var/lib/apt/lists <<EOF bash
    set -exuo pipefail
    apt-get update && apt-get install -y --no-install-recommends \
    python3-pip gdb
    echo "pre-commit" >> requirements.txt
    pip3 install --break-system-packages -r requirements.txt
EOF
