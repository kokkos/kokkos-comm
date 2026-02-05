// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <string>
#include <cstdlib>
#include <mpi.h>
#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>

// This reporter does nothing.
// We can use it to disable output from all but the root process
class NullReporter : public ::benchmark::BenchmarkReporter {
 public:
  NullReporter() {}
  virtual bool ReportContext(const Context &) { return true; }
  virtual void ReportRuns(const std::vector<Run> &) {}
  virtual void Finalize() {}
};

bool has_output_flag(int argc, char **argv) {
  for (int i = 0; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--benchmark_out=") == 0) {
      return true;
    }
  }
  return false;
}

bool has_output_envvar() { return std::getenv("BENCHMARK_OUT") != nullptr; }

// The main is rewritten to allow for MPI initializing and for selecting a
// reporter according to the process rank
int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    throw std::runtime_error("MPI_THREAD_MULTIPLE is needed");
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Kokkos::initialize();

  ::benchmark::Initialize(&argc, argv);

  if (rank == 0)
    // root process will use a reporter from the usual set provided by
    // ::benchmark
    ::benchmark::RunSpecifiedBenchmarks();
  else {
    // reporting from other processes is disabled by passing a custom reporter
    NullReporter null;
    bool has_file_output = has_output_flag(argc, argv) || has_output_envvar();
    if (has_file_output) {
      ::benchmark::RunSpecifiedBenchmarks(&null, &null);
    } else {
      ::benchmark::RunSpecifiedBenchmarks(&null);
    }
  }

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
