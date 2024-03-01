#include "test_utils.hpp"

#include "KokkosComm.hpp"

template <typename Space, typename View>
void send_recv(benchmark::State &, MPI_Comm comm, const Space &space, int rank,
               const View &v) {
  if (0 == rank) {
    KokkosComm::send(space, v, 1, 0, comm);
    KokkosComm::recv(space, v, 1, 0, comm);
  } else if (1 == rank) {
    KokkosComm::recv(space, v, 0, 0, comm);
    KokkosComm::send(space, v, 0, 0, comm);
  }
}

void benchmark_sendrecv(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    state.SkipWithError("benchmark_sendrecv needs at least 2 ranks");
  }

  using Scalar = double;

  auto space = Kokkos::DefaultExecutionSpace();
  using view_type = Kokkos::View<Scalar *>;
  view_type a("", 1000000);

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD,
                 send_recv<Kokkos::DefaultExecutionSpace, view_type>, space,
                 rank, a);
  }

  state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * a.size() * 2);
}

BENCHMARK(benchmark_sendrecv)->UseManualTime()->Unit(benchmark::kMillisecond);