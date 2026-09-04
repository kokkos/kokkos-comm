// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/fwd.hpp>
#include <KokkosComm/traits.hpp>
#include <KokkosComm/datatype.hpp>
#include "mpi_space.hpp"

#include "impl/error_handling.hpp"

namespace KokkosComm {

template <>
class Channel<MpiSpace> {
 public:
  explicit Channel(int dst_rank, int src_rank, int tag, MPI_Comm comm)
      : dst_rank_(dst_rank), src_rank_(src_rank), tag_(tag), comm_(comm) {}

  ~Channel() { release_requests(); }

  Channel(const Channel&)                    = delete;
  auto operator=(const Channel&) -> Channel& = delete;

  Channel(Channel&& other) noexcept
      : requests_(std::exchange(other.requests_, std::vector<MPI_Request>{})),
        statuses_(std::exchange(other.statuses_, std::vector<MPI_Status>{})),
        dst_rank_(other.dst_rank_),
        src_rank_(other.src_rank_),
        tag_(other.tag_),
        comm_(other.comm_) {}

  auto operator=(Channel&& other) noexcept -> Channel& {
    if (this != &other) {
      release_requests();
      requests_ = std::exchange(other.requests_, std::vector<MPI_Request>{});
      statuses_ = std::exchange(other.statuses_, std::vector<MPI_Status>{});
      dst_rank_ = other.dst_rank_;
      src_rank_ = other.src_rank_;
      tag_      = other.tag_;
      comm_     = other.comm_;
    }
    return *this;
  }

  // Registers a persistent send request
  template <class SendView>
  void sendinit(SendView view) {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::sendinit");
    using value_type = typename SendView::value_type;
    requests_.push_back(MPI_REQUEST_NULL);
    int err = MPI_Send_init(
        KokkosComm::data_handle(view), KokkosComm::span(view), datatype<MpiSpace, value_type>(), dst_rank_, tag_, comm_,
        &requests_.back()
    );
    mpi::fail_if(err != MPI_SUCCESS, "KokkosComm::Channel::sendinit: request initialization failed", comm_);
    statuses_.resize(requests_.size());
    Kokkos::Tools::popRegion();
  }

  // Registers a persistent receive request
  template <class RecvView>
  void recvinit(RecvView view) {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::recvinit");
    using value_type = typename RecvView::value_type;
    requests_.push_back(MPI_REQUEST_NULL);
    int err = MPI_Recv_init(
        KokkosComm::data_handle(view), KokkosComm::span(view), datatype<MpiSpace, value_type>(), src_rank_, tag_, comm_,
        &requests_.back()
    );
    mpi::fail_if(err != MPI_SUCCESS, "KokkosComm::Channel::recvinit: request initialization failed", comm_);
    statuses_.resize(requests_.size());
    Kokkos::Tools::popRegion();
  }

  void start() {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::start");
    Kokkos::fence();
    int err = MPI_Startall(static_cast<int>(requests_.size()), requests_.data());
    mpi::fail_if(err != MPI_SUCCESS, "KokkosComm::Channel::start: request start failed", comm_);
    Kokkos::Tools::popRegion();
  }

  void wait() {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::wait");
    int err = MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), statuses_.data());
    mpi::fail_if(err != MPI_SUCCESS, "KokkosComm::Channel::wait: request completion failed", comm_);
    Kokkos::Tools::popRegion();
  }

 private:
  auto release_requests() noexcept -> void {
    for (auto& request : requests_) {
      if (request != MPI_REQUEST_NULL) {
        int err = MPI_Request_free(&request);
        mpi::fail_if(err != MPI_SUCCESS, "KokkosComm::Channel: request release failed", comm_);
      }
    }
    requests_.clear();
    statuses_.clear();
  }

  std::vector<MPI_Request> requests_;  // Registered persistent requests
  std::vector<MPI_Status> statuses_;   // Completion statuses
  int dst_rank_;                       // Destination rank for send
  int src_rank_;                       // Source rank for receive
  int tag_;                            // MPI tag
  MPI_Comm comm_;                      // MPI communicator
};

}  // namespace KokkosComm
