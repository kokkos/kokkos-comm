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

// This example demonstrates how to perform a distributed matrix-vector product (A * x = y)
// using KokkosComm. The matrix A is distributed among the ranks in blocks of contiguous rows.
// Each rank owns a part of the vector x and will communicate it with other ranks step by step.
// At each step, a node communicates with two other nodes, one to receive the next part of x and
// the other that needs our part of x.

// Two helper objects are used to manage the distributed work:
// - RankDims: contains the number of rows, the start and end indices for each rank.
// - RankCommInfo: contains the ranks to send data to and receive data from.

// Other communication patterns can be used to exchange the vector x between ranks.

#include <KokkosComm/KokkosComm.hpp>

#include <iostream>
#include <string_view>

struct RankDims {
  int nb_rows;
  int row_start;
  int row_end;

  RankDims(int N, int rank, int size) {
    this->nb_rows   = N / size;
    this->row_start = rank * this->nb_rows;
    this->row_end   = (rank + 1) * this->nb_rows;

    if (rank == size - 1) {
      this->row_end = N;
      this->nb_rows = this->row_end - this->row_start;
    }
  }
};

struct RankCommInfo {
  Kokkos::View<int*> recv_ranks;
  Kokkos::View<int*> send_ranks;

  RankCommInfo(int rank, int size) {
    this->recv_ranks = Kokkos::View<int*>("recv_ranks", size);
    this->send_ranks = Kokkos::View<int*>("send_ranks", size);

    for (int i = 0; i < size; i++) {
      this->recv_ranks(i) = (rank + i) % size;
      this->send_ranks(i) = (rank - i + size) % size;
    }
  }
};

template <typename View>
bool verify_result(View& y, int N) {
  const double N_d = static_cast<double>(N);
  const double sum = N_d * (N_d + 1) * (2 * N_d + 1) / 6;
  int errors       = 0;

  Kokkos::parallel_reduce("Verify", y.extent(0), KOKKOS_LAMBDA(const int i, int& err) {
    err += (y(i) != sum);
  }, errors);

  return (errors == 0);
}

int main(int argc, char* argv[]) {
  int N = 1 << 12;
  int M = 1 << 12;

  for (int i = 1; i < argc; i++) {
    std::string_view arg(argv[i]);

    if (arg == "-h") {
      std::cout << "KokkosComm dense matrix-vector product example \n"
                << "Options: \n"
                << " [-M <size>] Number of rows in matrix A (default: 2^12)\n"
                << " [-N <size>] Number of columns in A (default: 2^12)\n"
                << " [-h]        for help" << std::endl;
      return 0;
    } else if (arg == "-N" && argc > i + 1) {
      N = std::stoi(argv[i+1]);
    } else if (arg == "-M" && argc > i + 1) {
      M = std::stoi(argv[i+1]);
    }
  }

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "MPI_Init failed" << std::endl;
    return 1;
  }

  Kokkos::initialize(argc, argv);
  {
  using ExecSpace   = Kokkos::DefaultExecutionSpace;
  using CommSpace   = KokkosComm::Mpi;
  using MatrixType  = Kokkos::View<double**, ExecSpace>;
  using VectorType  = Kokkos::View<double*, ExecSpace>;
  using KKPair      = Kokkos::pair<int, int>;

  int rank, size;
  KokkosComm::Handle<ExecSpace, CommSpace> handle;
  rank = handle.rank();
  size = handle.size();

  if (rank == 0) {
    std::cout << "Running with " << size << " ranks\n"
              << "Matrix size " << M << " x " << N << std::endl;
  }

  // Helper objects for distributed work
  RankDims dim_mtx(M, rank, size);
  RankDims dim_vec(N, rank, size);
  RankCommInfo comm_info(rank, size);

  MatrixType A("A", dim_mtx.nb_rows, N);
  VectorType x("x", dim_vec.nb_rows);
  VectorType y("y", dim_mtx.nb_rows);

  // Initialize A and x
  Kokkos::parallel_for("Initialize", dim_mtx.nb_rows, KOKKOS_LAMBDA(const int i) {
    for (int j = 0; j < N; j++) {
      A(i, j) = j + 1.0;
    }
  });

  Kokkos::parallel_for("Initialize x", dim_vec.nb_rows, KOKKOS_LAMBDA(const int i) {
    x(i) = dim_vec.row_start + i +  1.0;
  });

  // Set communication vector for send/recv operations
  VectorType comm_vector("comm_vector", dim_vec.nb_rows + size);
  VectorType compute_vector("computed_vector", dim_vec.nb_rows + size);

  // Span for the computation on the current step
  auto span_compute = Kokkos::subview(compute_vector, KKPair(0, dim_vec.nb_rows));
  // Span to store the received data for next step
  auto span_comm = Kokkos::subview(comm_vector, KKPair(0, dim_vec.nb_rows + size));

  // Copy x to the computation vector for first step
  Kokkos::deep_copy(span_compute, x);
  RankDims current_dim = dim_vec;

  // Communication and computation steps
  for (int step = 1; step < size; step++) {
    int send_comm_rank = comm_info.send_ranks(step);
    int recv_comm_rank = comm_info.recv_ranks(step);

    // Prepare to receive data
    RankDims next_dim(N, recv_comm_rank, size);
    span_comm = Kokkos::subview(comm_vector, KKPair(0, next_dim.nb_rows));

    // Start MPI communication (non-blocking)
    auto req_send = KokkosComm::send(handle, x, send_comm_rank);
    auto req_recv = KokkosComm::recv(handle, span_comm, recv_comm_rank);

    // Compute with current data while communication may happen in the background
    Kokkos::parallel_for("MatrixVectorProduct", dim_mtx.nb_rows, KOKKOS_LAMBDA(const int i) {
      for (int j = 0; j < current_dim.nb_rows; j++) {
        y(i) += A(i, j + current_dim.row_start) * span_compute(j);
      }
    });

    // Wait for the communication to finish
    KokkosComm::wait(req_send);
    KokkosComm::wait(req_recv);

    // Copy the received data to span_compute for the next step
    span_compute = Kokkos::subview(compute_vector, KKPair(0, next_dim.nb_rows));
    Kokkos::deep_copy(span_compute, span_comm);
    current_dim = next_dim;
  }

  // Last step
  Kokkos::parallel_for("MatrixVectorProduct tail", dim_mtx.nb_rows, KOKKOS_LAMBDA(const int i) {
    for (int j = 0; j < current_dim.nb_rows; j++) {
      y(i) += A(i, j + current_dim.row_start) * span_compute(j);
    }
  });

  // Wait for all nodes
  KokkosComm::mpi::barrier(handle.mpi_comm());

  // Check the result
  bool success = verify_result(y, N);
  if (!success) {
    std::cout << "Rank " << rank << " failed to verify the result" << std::endl;
  }

  }  // Finalize MPI and Kokkos
  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
