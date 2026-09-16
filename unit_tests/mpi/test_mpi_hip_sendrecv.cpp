// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/*
This test verifies that GPU-aware MPI is operating as expected if HIP is enabled.
If not, the HIP error is specifically reported using hipGetErrorString().
*/

#include <iostream>

#include <mpi.h>
#include <hip/hip_runtime.h>

// Macro to check for HIP errors
#define HIP(call)                                                                                                  \
  do {                                                                                                             \
    hipError_t err = call;                                                                                         \
    if (err != hipSuccess) {                                                                                       \
      std::cerr << "HIP error in file '" << __FILE__ << "' in line " << __LINE__ << ": " << hipGetErrorString(err) \
                << " (" << err << ")" << std::endl;                                                                \
      exit(EXIT_FAILURE);                                                                                          \
    }                                                                                                              \
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

  HIP(hipMalloc(&d_recv_buf, buffer_size));
  HIP(hipMalloc(&d_errs, sizeof(int)));
  HIP(hipMemset(d_errs, 0, sizeof(int)));
  Scalar* recv_buf = reinterpret_cast<Scalar*>(byte_offset(d_recv_buf, alignment));

  if (rank == 0) {
    Scalar* d_send_buf;
    HIP(hipMalloc(&d_send_buf, buffer_size));
    Scalar* send_buf = reinterpret_cast<Scalar*>(byte_offset(d_send_buf, alignment));
    init_array<<<(num_elements + 255) / 256, 256>>>(send_buf, num_elements);
    HIP(hipDeviceSynchronize());
    MPI_Send(send_buf, num_elements, mpi_type<Scalar>(), 1, 0, MPI_COMM_WORLD);
    HIP(hipFree(d_send_buf));
  } else if (rank == 1) {
    MPI_Recv(recv_buf, num_elements, mpi_type<Scalar>(), 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    check_array<<<(num_elements + 255) / 256, 256>>>(recv_buf, num_elements, d_errs);
    HIP(hipDeviceSynchronize());
  }

  HIP(hipMemcpy(&h_errs, d_errs, sizeof(int), hipMemcpyDeviceToHost));

  if (h_errs > 0) {
    std::cerr << "[" << rank << "] " << __FILE__ << ":" << __LINE__ << " h_errs=" << h_errs << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  HIP(hipFree(d_recv_buf));
  HIP(hipFree(d_errs));
}

template <typename Scalar>
void run_test() {
  int offset = 128;
  for (size_t _ : {0, 1, 2}) {  // run a few times
    for (size_t n : {1000, 113, 16, 8, 4, 2, 1}) {
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
