//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

/*
https://github.com/pmodels/mpich/issues/7304
This test may run incorrectly in MPICH 4.2.3 and 4.3.0, and possibly other versions, configured with OFI/libfabrics.
Try something like:
spack install mpich@4.2.3 +cuda cuda_arch=86 device=ch4 netmod=ofi
This test attempts to reproduce an issue where a two-rank CUDA-aware MPI_Send / MPI_Recv combo was observed to either
produce incorrect data after MPI_Recv or trigger an internal error in MPICH 4.2.3 and 4.3.0. There are two unusual
things about this test:
- The recv buffer is allocated on both the send and recv side (only used on recv side), while the send buffer is only
allocated on the send side. This is because reproducing the error was sensitive to the allocations happening, and this
was one way to trigger it.
- There is an optional "alignment" of the buffer, where the starting offset passed to MPI_Send / MPI_Recv is not 0. This
was an attempt to mimic the fact that Kokkos Views have some metadata allocated before the actual data. It turns out the
test fails with offset=0 and offset=128 in different ways.

The error was determined to arise because CUDA was linked before the MPI libraries, which prevented MPICH's hooked
implementations from being used. Also, the CUDA libraries were statically linked.
*/

#include <iostream>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Macro to check for CUDA errors
#define CUDA(call)                                                                                                   \
  do {                                                                                                               \
    cudaError_t err = call;                                                                                          \
    if (err != cudaSuccess) {                                                                                        \
      std::cerr << "CUDA error in file '" << __FILE__ << "' in line " << __LINE__ << ": " << cudaGetErrorString(err) \
                << " (" << err << ")" << std::endl;                                                                  \
      exit(EXIT_FAILURE);                                                                                            \
    }                                                                                                                \
  } while (0)

namespace {

template <typename Scalar>
__global__ void init_array(Scalar* a, int sz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < sz) {
    a[i] = Scalar(i);
  }
}

template <typename Scalar>
__global__ void check_array(const Scalar* a, int sz, int* errs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < sz && a[i] != Scalar(i)) {
    atomicAdd(errs, 1);
    printf("ERROR: a[%d](%p) = %f != %f\n", int(i), a + i, double(a[i]), double(i));
  }
}

// get the built-in MPI Datatype for int32_t, int64_t, or float
template <typename Scalar>
MPI_Datatype mpi_type() {
  if constexpr (std::is_same_v<Scalar, int32_t>) {
    return MPI_INT;
  } else if constexpr (std::is_same_v<Scalar, int64_t>) {
    return MPI_LONG_LONG;
  } else if constexpr (std::is_same_v<Scalar, float>) {
    return MPI_FLOAT;
  } else {
    static_assert(std::is_void_v<Scalar>, "unsupported type");
  }
}

// return ptr + offset (in bytes)
void* byte_offset(void* ptr, std::size_t offset) {
  return reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(ptr) + offset);
}

template <typename Scalar>
void run_test(int num_elements, int alignment) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (0 == rank) {
    // get a string name of the Scalar type
    const char* name;
    if constexpr (std::is_same_v<Scalar, int32_t>) {
      name = "int32_t";
    } else if constexpr (std::is_same_v<Scalar, float>) {
      name = "float";
    } else if constexpr (std::is_same_v<Scalar, int64_t>) {
      name = "int64_t";
    } else {
      static_assert(std::is_void_v<Scalar>, "unsupported type");
    }

    std::cerr << __FILE__ << ":" << __LINE__ << " test: " << num_elements << " " << name << " " << alignment << "\n";
  }

  if (2 != size) {
    std::cerr << "test requires 2 processes, got " << size << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  Scalar* d_recv_buf;
  int* d_errs;
  int h_errs = 0;

  size_t buffer_size = num_elements * sizeof(Scalar) + alignment;

  CUDA(cudaMalloc(&d_recv_buf, buffer_size));
  CUDA(cudaMalloc(&d_errs, sizeof(int)));
  CUDA(cudaMemset(d_errs, 0, sizeof(int)));
  Scalar* recv_buf = reinterpret_cast<Scalar*>(byte_offset(d_recv_buf, alignment));

  if (rank == 0) {
    Scalar* d_send_buf;
    CUDA(cudaMalloc(&d_send_buf, buffer_size));
    Scalar* send_buf = reinterpret_cast<Scalar*>(byte_offset(d_send_buf, alignment));
    init_array<<<(num_elements + 255) / 256, 256>>>(send_buf, num_elements);
    CUDA(cudaDeviceSynchronize());
    MPI_Send(send_buf, num_elements, mpi_type<Scalar>(), 1, 0, MPI_COMM_WORLD);
    CUDA(cudaFree(d_send_buf));
  } else if (rank == 1) {
    MPI_Recv(recv_buf, num_elements, mpi_type<Scalar>(), 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    check_array<<<(num_elements + 255) / 256, 256>>>(recv_buf, num_elements, d_errs);
    CUDA(cudaDeviceSynchronize());
  }

  CUDA(cudaMemcpy(&h_errs, d_errs, sizeof(int), cudaMemcpyDeviceToHost));

  if (h_errs > 0) {
    std::cerr << "[" << rank << "] " << __FILE__ << ":" << __LINE__ << " h_errs=" << h_errs << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  CUDA(cudaFree(d_recv_buf));
  CUDA(cudaFree(d_errs));
}

template <typename Scalar>
void run_test() {
  int offset = 128;
  for (size_t _ : {0, 1, 2}) {  // run a few times
    for (size_t n : {113, 16, 8, 4, 2, 1}) {
      MPI_Barrier(MPI_COMM_WORLD);
      run_test<Scalar>(n, offset);
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  run_test<int32_t>();
  run_test<int64_t>();
  run_test<float>();
  MPI_Finalize();
  return 0;
}
