#include <memory>
#include <mutex>

#include <mpi.h>

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

std::unique_ptr<XcclCtx> XcclCtx::instance_{};
std::once_flag XcclCtx::init_flag_{};

XcclCtx::XcclCtx(ncclComm_t comm, DeviceStream stream, int dev, int size, int rank)
    : comm_(comm), stream_(stream), dev_(dev), size_(size), rank_(rank) {}

XcclCtx::~XcclCtx() {
  if (stream_ != nullptr) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
    cudaStreamDestroy(stream_);
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
    hipStreamDestroy(stream_);
#endif
  }
  if (comm_ != nullptr) {
    ncclCommDestroy(comm_);
  }
}

auto XcclCtx::init(bool verbose) -> void {
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
#if defined(KOKKOSCOMM_ENABLE_NCCL)
    KC_CUDA_CHECK(cudaGetDeviceCount(&devs));
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
    KC_HIP_CHECK(hipGetDeviceCount(&devs));
#endif

    if (verbose) {
      KC_INFO("P{} found {} GPU devices", rank, devs);
    }

    KC_CHECK(local_rank < devs, "P{} needs device #{} but only {} devices available", rank, local_rank, devs);

#if defined(KOKKOSCOMM_ENABLE_NCCL)
    KC_CUDA_CHECK(cudaSetDevice(local_rank));
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
    KC_HIP_CHECK(hipSetDevice(local_rank));
#endif

    if (verbose) {
      KC_INFO("P{} assigned to GPU device #{}", rank, local_rank);
    }

    ncclUniqueId nccl_id{};
    if (rank == 0) {
      KC_XCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }

    KC_MPI_CHECK(MPI_Bcast(&nccl_id, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, mpi_comm));

    ncclComm_t nccl_comm = nullptr;
    KC_XCCL_CHECK(ncclCommInitRank(&nccl_comm, size, nccl_id, rank));

#if defined(KOKKOSCOMM_ENABLE_NCCL)
    DeviceStream stream = nullptr;
    KC_CUDA_CHECK(cudaStreamCreate(&stream));
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
    DeviceStream stream = nullptr;
    KC_HIP_CHECK(hipStreamCreate(&stream));
#endif

    instance_ = std::unique_ptr<XcclCtx>(new XcclCtx(nccl_comm, stream, local_rank, size, rank));
  });
}

auto XcclCtx::fini() -> void { instance_.reset(); }

auto XcclCtx::get() -> XcclCtx& {
  KC_CHECK(instance_ != nullptr, "xCCL context not initialized");
  return *instance_;
}

auto XcclCtx::comm() const -> ncclComm_t { return comm_; }

auto XcclCtx::stream() const -> DeviceStream { return stream_; }

auto XcclCtx::size() const -> int { return size_; }

auto XcclCtx::rank() const -> int { return rank_; }

auto XcclCtx::device() const -> int { return dev_; }

}  // namespace test_utils
