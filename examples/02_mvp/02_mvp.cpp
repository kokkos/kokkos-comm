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

// This example demonstrates how to perform a distributed matrix-vector product
// (A * x = y) using KokkosComm. The matrix A is distributed in 2D blocks across
// a process grid. first step is column-wise communication for distributed matrix-vector
// product, and the second step is row-wise communication for summing partial results.
// The vector x is distributed to correspond to the matrix columns.

#include <KokkosComm/KokkosComm.hpp>

#include <cmath>
#include <cstdio>

#include <iostream>
#include <string_view>

struct RankDimsMtx {
  int num_rows;
  int num_cols;
  int row_offset;
  int col_offset;

  RankDimsMtx(int M, int N, int rank, int size) {
    const int grid_size = static_cast<int>(std::sqrt(size));

    const int block_col = rank / grid_size;
    const int block_row = rank % grid_size;

    this->row_offset = block_row * (M / grid_size);
    this->col_offset = block_col * (N / grid_size);
    int row_end = (block_row + 1) * (M / grid_size);
    int col_end = (block_col + 1) * (N / grid_size);

    if (block_row == grid_size - 1) {
      row_end = M;
    }
    if (block_col == grid_size - 1) {
      col_end = N;
    }
    this->num_rows = row_end - this->row_offset;
    this->num_cols = col_end - this->col_offset;
  }
};

// Split the vector into blocks to correspond to the matrix columns
// and then split the blocks to correspond to the matrix rows
struct RankDimsVec {
  int segment_size;
  int row_offset;

  RankDimsVec(int N, int rank, int size) {
    const int grid_size = static_cast<int>(std::sqrt(size));
    const int block_col = rank / grid_size;
    const int block_row = rank % grid_size;

    int block_row_offset = block_col * (N / grid_size);
    int block_row_end    = (block_col + 1) * (N / grid_size);
    if (block_col == grid_size - 1) {
      block_row_end = N;
    }
    const int block_size = block_row_end - block_row_offset;

    this->row_offset = block_row_offset + block_row * (block_size / grid_size);
    int row_end      = block_row_offset + (block_row + 1) * (block_size / grid_size);
    if (block_row == grid_size - 1) {
      row_end = block_row_end;
    }
    this->segment_size = row_end - this->row_offset;
  }
};

struct CommPattern {
  Kokkos::View<int*> recv_ranks;
  Kokkos::View<int*> send_ranks;

  CommPattern(int rank, int size) {
    this->recv_ranks = Kokkos::View<int*>("recv_ranks", size);
    this->send_ranks = Kokkos::View<int*>("send_ranks", size);

    for (int i = 0; i < size; i++) {
      this->recv_ranks(i) = (rank + i) % size;
      this->send_ranks(i) = (rank - i + size) % size;
    }
  }
};

template <typename View>
bool verifyResults(View& y, int N) {
  const double N_d      = static_cast<double>(N);
  const double expected = N_d * (N_d + 1) * (2 * N_d + 1) / 6;
  int errors            = 0;

  Kokkos::parallel_reduce("Validate", y.extent(0), KOKKOS_LAMBDA(const int i, int& err) {
    err += (y(i) != expected);
  }, errors);

  return (errors == 0);
}

int main(int argc, char* argv[]) {
  int M = 1 << 12;
  int N = 1 << 12;

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
      N = std::stoi(argv[i + 1]);
    } else if (arg == "-M" && argc > i + 1) {
      M = std::stoi(argv[i + 1]);
    }
  }

  // Initialize MPI
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "MPI_Init failed" << std::endl;
    return 1;
  }

  Kokkos::initialize(argc, argv);
  {
  using ExecSpace  = Kokkos::DefaultExecutionSpace;
  using CommSpace  = KokkosComm::Mpi;
  using MatrixType = Kokkos::View<double**, ExecSpace>;
  using VectorType = Kokkos::View<double*, ExecSpace>;
  using IndexRange = Kokkos::pair<int, int>;
  using CommHandle = KokkosComm::Handle<ExecSpace, CommSpace>;

  CommHandle globalHandle;
  const int worldRank = globalHandle.rank();
  const int worldSize = globalHandle.size();

  const int gridSize = static_cast<int>(std::sqrt(worldSize));
  if (gridSize * gridSize != worldSize) {
    if (worldRank == 0) {
      std::cerr << "Error: number of processes (" << worldSize
                << ") must be a perfect square" << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  if (worldRank == 0) {
    std::cout << "Running with " << worldSize << " ranks\n"
              << "Matrix size " << M << " x " << N << std::endl;
  }

  // Create column communicator
  MPI_Comm colComm;
  int col = worldRank / gridSize;
  MPI_Comm_split(MPI_COMM_WORLD, col, worldRank, &colComm);

  // Create row communicator
  MPI_Comm rowComm;
  int row = worldRank % gridSize;
  MPI_Comm_split(MPI_COMM_WORLD, row, worldRank, &rowComm);

  // Create KokkosComm handles for each communicator
  CommHandle colHandle(colComm);
  CommHandle rowHandle(rowComm);

  // Transpose the rank mapping
  RankDimsVec vectorDist(N, worldRank, worldSize);
  RankDimsMtx matrixDist(M, N, worldRank, worldSize);

  MatrixType A("A", matrixDist.num_rows, matrixDist.num_cols);
  VectorType x("x", vectorDist.segment_size);
  VectorType y("y", matrixDist.num_rows);

  // Initialize A and x
  Kokkos::parallel_for("Initialize", matrixDist.num_rows, KOKKOS_LAMBDA(const int i) {
    for (int j = 0; j < matrixDist.num_cols; j++) {
      A(i, j) = matrixDist.col_offset + j + 1.0;
    }
  });

  Kokkos::parallel_for("Initialize", vectorDist.segment_size, KOKKOS_LAMBDA(const int i) {
    x(i) = vectorDist.row_offset + i + 1.0;
  });

  // Allocates buffers for communication and computation
  VectorType recvBuffer("ReceiveBuffer", matrixDist.num_cols);
  VectorType workBuffer("WorkBuffer", matrixDist.num_cols);

  // Copy x to the computation vector for first step
  auto workSpan = Kokkos::subview(workBuffer, IndexRange(0, vectorDist.segment_size));
  Kokkos::deep_copy(workSpan, x);

  CommPattern commPattern(colHandle.rank(), colHandle.size());
  RankDimsVec currentDist = vectorDist;

  // Computation in each column is independent
  for (int step = 1; step < gridSize; step++) {
    const int sendRank = commPattern.send_ranks(step);
    const int recvRank = commPattern.recv_ranks(step);

    // Get the global rank for the dimension
    const int recvGlobalRank = (worldRank / gridSize) * gridSize + recvRank;
    RankDimsVec nextDist(N, recvGlobalRank, worldSize);

    auto recvSpan = Kokkos::subview(recvBuffer, IndexRange(0, nextDist.segment_size));
    auto reqSend = KokkosComm::send(colHandle, x, sendRank);
    auto reqRecv = KokkosComm::recv(colHandle, recvSpan, recvRank);

    // Compute with current data while communication may happen in the background
    Kokkos::parallel_for("MatrixVectorProduct", matrixDist.num_rows, KOKKOS_LAMBDA(const int i) {
      for (int j = 0; j < currentDist.segment_size; j++) {
        y(i) += A(i, j + currentDist.row_offset - matrixDist.col_offset) * workSpan(j);
      }
    });

    // Wait for the communication to finish
    KokkosComm::wait(reqSend);
    KokkosComm::wait(reqRecv);

    workSpan = Kokkos::subview(workBuffer, IndexRange(0, nextDist.segment_size));

    // Copy the received data to span_compute for the next step
    Kokkos::deep_copy(workSpan, recvSpan);
    currentDist = nextDist;
  }

  // Last step
  Kokkos::parallel_for("MatrixVectorProduct", matrixDist.num_rows, KOKKOS_LAMBDA(const int i) {
    for (int j = 0; j < currentDist.segment_size; j++) {
      y(i) += A(i, j + currentDist.row_offset - matrixDist.col_offset) * workSpan(j);
    }
  });

  recvBuffer = VectorType("RecvBuffer", matrixDist.num_rows);
  workBuffer = VectorType("WorkBuffer", matrixDist.num_rows);
  VectorType partialResult("partialResult", matrixDist.num_rows);

  Kokkos::deep_copy(partialResult, y);
  Kokkos::deep_copy(workBuffer, partialResult);
  Kokkos::deep_copy(y, 0.0);

  commPattern = CommPattern(rowHandle.rank(), rowHandle.size());

  // Summation over the rows
  for (int sum_step = 1; sum_step < gridSize; sum_step++) {
    const int sendRank = commPattern.send_ranks(sum_step);
    const int recvRank = commPattern.recv_ranks(sum_step);

    auto reqSend = KokkosComm::send(rowHandle, partialResult, sendRank);
    auto reqRecv = KokkosComm::recv(rowHandle, recvBuffer, recvRank);

    Kokkos::parallel_for("Sum", matrixDist.num_rows, KOKKOS_LAMBDA(const int i) {
      y(i) += workBuffer(i);
    });

    // Wait for the communication to finish
    KokkosComm::wait(reqSend);
    KokkosComm::wait(reqRecv);

    // Copy the received data to span_compute for the next step
    Kokkos::deep_copy(workBuffer, recvBuffer);
  }

  // Last step
  Kokkos::parallel_for("Sum", matrixDist.num_rows, KOKKOS_LAMBDA(const int i) {
    y(i) += workBuffer(i);
  });

  // Check the result
  bool success = verifyResults(y, N);
  Kokkos::View<int[1]> error("errors");
  Kokkos::View<int*> errors("errors", worldSize);
  error(0) = !success;

  KokkosComm::mpi::allgather(error, errors, globalHandle.mpi_comm());

  if (worldRank == 0) {
    int total_errors = 0;
    for (int i = 0; i < worldSize; i++) {
      if (errors(i) != 0) {
        std::cout << "Rank " << i << " failed to verify the result" << std::endl;
        total_errors++;
      }
    }
    if (total_errors == 0) {
      std::cout << "All ranks verified the result" << std::endl;
    }
  }

  MPI_Comm_free(&colComm);
  MPI_Comm_free(&rowComm);

  }  // Finalize MPI and Kokkos
  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
