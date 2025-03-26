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
// using KokkosComm. The matrix A is distributed among the ranks by blocks of contiguous rows.
// Each rank owns a part of the vector x and will communicate it with other ranks step by step.
// At each step a node communicates with two other nodes to send and receive data.

// Two helper objects are used to manage the distributed work:
// - RankDims: contains the number of rows, the start and end row indices for each rank
// - RankCommInfo: contains the ranks to send data to and receive data from

// Other algorithms can be used to distribute the work.

#include <KokkosComm/KokkosComm.hpp>
#include <iostream>

struct RankDims {
  unsigned int nb_rows;
  unsigned int row_start;
  unsigned int row_end;

  RankDims(long N, int rank, int size) {
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
  Kokkos::View<unsigned int*> recv_ranks;
  Kokkos::View<unsigned int*> send_ranks;

  RankCommInfo(int rank, int size) {
    this->recv_ranks = Kokkos::View<unsigned int*>("recv_ranks", size);
    this->send_ranks = Kokkos::View<unsigned int*>("send_ranks", size);

    for (int i = 0; i < size; i++) {
      this->recv_ranks(i) = (rank + i) % size;
      this->send_ranks(i) = (rank - i + size) % size;
    }
  }
};

// Provide a default value for N
void checkArgs(long& N) {
  if (N < 1) {
    N = 1 << 12;
  }
}

template <typename View>
bool verify_result(View& y, long N) {
  for (unsigned long i = 0; i < y.extent(0); i++) {
    if (y(i) != N) {
      return false;
    }
  }
  return true;
}

int main(int argc, char* argv[]) {
  long N = -1;

  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) {
      N = std::atoi(argv[++i]);
    }
    if (strcmp(argv[i], "-h") == 0) {
      std::cout << "KokkosComm dense square matrix-vector product example \n"
                << "  Usage: " << argv[0] << " [-N <size>] default size is 2^12" << std::endl;
      return 0;
    }
  }
  checkArgs(N);

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    throw std::runtime_error("MPI_THREAD_MULTIPLE is needed");
  }

  Kokkos::initialize(argc, argv);
  {
  using ExecSpace   = Kokkos::DefaultExecutionSpace;
  using CommSpace   = KokkosComm::DefaultCommunicationSpace;
  using matrix_type = Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace>;
  using vector_type = Kokkos::View<double*, Kokkos::LayoutRight, ExecSpace>;
  using kk_pair     = Kokkos::pair<long, long>;

  int rank, size;
  KokkosComm::Handle<ExecSpace, CommSpace> handle;
  rank = handle.rank();
  size = handle.size();

  // Helper objects for distributed work
  RankDims dim(N, rank, size);
  RankCommInfo comm_info(rank, size);

  matrix_type A("A", dim.nb_rows, N);
  vector_type x("x", dim.nb_rows);
  vector_type y("y", dim.nb_rows);

  // Initialize A, x, y
  Kokkos::parallel_for("Initialize", dim.nb_rows, KOKKOS_LAMBDA(const int i) {
    for (int j = 0; j < N; j++) {
      A(i, j) = 1.0;
    }
    x(i) = 1.0;
    y(i) = 0.0;
  });

  // Set communication vector for send/recv operations
  Kokkos::View<double*, Kokkos::LayoutRight, ExecSpace> comm_vector("comm_vector", dim.nb_rows + size);
  Kokkos::View<double*, Kokkos::LayoutRight, ExecSpace> compute_vector("computed_vector", dim.nb_rows + size);

  // Span for the computation on the current step
  auto span_compute = Kokkos::subview(compute_vector, kk_pair(0, dim.nb_rows));
  // Span to store the received data for next step
  auto span_comm = Kokkos::subview(comm_vector, kk_pair(0, dim.nb_rows + size));

  // Copy x to the computation vector for first step
  Kokkos::deep_copy(span_compute, x);
  RankDims current_dim = dim;

  // Communication and computation steps
  for (int step = 1; step < size; step++) {
    int send_comm_rank = comm_info.send_ranks(step);
    int recv_comm_rank = comm_info.recv_ranks(step);

    // Prepare to receive data
    RankDims next_dim(N, recv_comm_rank, size);
    span_comm = Kokkos::subview(comm_vector, kk_pair(0, next_dim.nb_rows));

    // Start MPI communication (non-blocking)
    auto req_send = KokkosComm::send(handle, x, send_comm_rank);
    auto req_recv = KokkosComm::recv(handle, span_comm, recv_comm_rank);

    // Compute with current data while communication may happen in the background
    Kokkos::parallel_for("MatrixVectorProduct", dim.nb_rows, KOKKOS_LAMBDA(const int i) {
      for (unsigned int j = 0; j < current_dim.nb_rows; j++) {
        y(i) += A(i, j + current_dim.row_start) * span_compute(j);
      }
    });

    // Wait for the communication to finish
    KokkosComm::wait_all(req_send, req_recv);

    // Copy the received data to span_compute for the next step
    span_compute = Kokkos::subview(compute_vector, kk_pair(0, next_dim.nb_rows));
    Kokkos::deep_copy(span_compute, span_comm);
    current_dim = next_dim;
  }

  // Last step
  Kokkos::parallel_for("MatrixVectorProduct tail", dim.nb_rows, KOKKOS_LAMBDA(const int i) {
    for (unsigned int j = 0; j < current_dim.nb_rows; j++) {
      y(i) += A(i, j + current_dim.row_start) * span_compute(j);
    }
  });

  // Wait for all nodes
  KokkosComm::mpi::barrier(handle.mpi_comm());

  // Check the result
  if (verify_result(y, N)) {
    std::cout << "A * x = y computation successful" << std::endl;
  }

  // Finalize MPI and Kokkos
  } Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
