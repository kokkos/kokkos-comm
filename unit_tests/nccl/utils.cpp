#include <memory>
#include <mutex>

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

#include "utils.hpp"
#include "../logging.hpp"

namespace {

[[nodiscard]] auto get_local_rank(MPI_Comm comm, int my_rank) -> int {
  MPI_Comm node_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &node_comm);

  int node_rank;
  MPI_Comm_rank(node_comm, &node_rank);

  MPI_Comm_free(&node_comm);
  return node_rank;
}

}  // namespace

namespace test_utils {

std::unique_ptr<NcclCtx> NcclCtx::instance_{};
std::once_flag NcclCtx::init_flag_{};

NcclCtx::NcclCtx(ncclComm_t comm, cudaStream_t stream, int dev, int size, int rank)
    : comm_(comm), stream_(stream), dev_(dev), size_(size), rank_(rank) {}

NcclCtx::~NcclCtx() {
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
  if (comm_ != nullptr) {
    ncclCommDestroy(comm_);
  }
}

auto NcclCtx::init(bool verbose) -> void {
  std::call_once(init_flag_, [verbose]() {
    int flag = 0;
    KC_MPI_CHECK(MPI_Initialized(&flag));
    KC_CHECK(flag != 0, "MPI is not initialized");

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    int size = 0;
    KC_MPI_CHECK(MPI_Comm_size(mpi_comm, &size));
    int rank = 0;
    KC_MPI_CHECK(MPI_Comm_rank(mpi_comm, &rank));

    int local_rank = get_local_rank(mpi_comm, rank);

    int devs = 0;
    KC_CUDA_CHECK(cudaGetDeviceCount(&devs));

    if (verbose) {
      KC_INFO("P{} found {} CUDA devices", rank, devs);
    }

    KC_CHECK(local_rank < devs, "P{} needs device #{} but only {} devices available", rank, local_rank, devs);

    KC_CUDA_CHECK(cudaSetDevice(local_rank));

    if (verbose) {
      KC_INFO("P{} assigned to CUDA device #{}", rank, local_rank);
    }

    ncclUniqueId nccl_id{};
    if (rank == 0) {
      KC_NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }

    KC_MPI_CHECK(MPI_Bcast(&nccl_id, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, mpi_comm));

    ncclComm_t nccl_comm = nullptr;
    KC_NCCL_CHECK(ncclCommInitRank(&nccl_comm, size, nccl_id, rank));

    cudaStream_t stream = nullptr;
    KC_CUDA_CHECK(cudaStreamCreate(&stream));

    instance_ = std::unique_ptr<NcclCtx>(new NcclCtx(nccl_comm, stream, local_rank, size, rank));
  });
}

auto NcclCtx::fini() -> void { instance_.reset(); }

auto NcclCtx::get() -> NcclCtx& {
  KC_CHECK(instance_ != nullptr, "NCCL context not initialized");
  return *instance_;
}

auto NcclCtx::comm() const -> ncclComm_t { return comm_; }

auto NcclCtx::stream() const -> cudaStream_t { return stream_; }

auto NcclCtx::size() const -> int { return size_; }

auto NcclCtx::rank() const -> int { return rank_; }

auto NcclCtx::device() const -> int { return dev_; }

}  // namespace test_utils
