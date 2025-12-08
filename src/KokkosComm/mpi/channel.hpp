// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <Kokkos_Core.hpp>

#include <KokkosComm/traits.hpp>
#include "req.hpp"

#include "impl/types.hpp"

namespace KokkosComm {

template <typename CommSpace = DefaultCommunicationSpace>
class Channel {
 public:
  explicit Channel(int dest_rank, int src_rank, int tag, MPI_Comm comm)
      : dest_rank_(dest_rank), src_rank_(src_rank), tag_(tag), comm_(comm) {}

  // Send initialization - dynamically adds a send request to the send queue
  template <class SendView>
  void sendinit(SendView view) {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::sendinit");
    using value_type = typename SendView::value_type;
    // Add a new request to the send_reqs_ vector
    send_reqs_.emplace_back();
    MPI_Send_init(KokkosComm::data_handle(view), KokkosComm::span(view), KokkosComm::Impl::mpi_type_v<value_type>,
                  dest_rank_, tag_, comm_, &(send_reqs_.back().mpi_request()));
    Kokkos::Tools::popRegion();
  }

  // Receive initialization - dynamically adds a receive request to the recv queue
  template <class RecvView>
  void recvinit(RecvView view) {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::recvinit");
    using value_type = typename RecvView::value_type;
    recv_reqs_.emplace_back();
    MPI_Recv_init(KokkosComm::data_handle(view), KokkosComm::span(view), KokkosComm::Impl::mpi_type_v<value_type>,
                  src_rank_, tag_, comm_, &(recv_reqs_.back().mpi_request()));
    Kokkos::Tools::popRegion();
  }

  void start() {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::start");
    std::vector<MPI_Request> mpi_reqs;
    for (auto& req : send_reqs_) {
      mpi_reqs.push_back(req.mpi_request());
    }
    for (auto& req : recv_reqs_) {
      mpi_reqs.push_back(req.mpi_request());
    }
    Kokkos::fence();
    MPI_Startall(mpi_reqs.size(), mpi_reqs.data());
    Kokkos::Tools::popRegion();
  }

  void wait() {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::wait");
    std::vector<Req<MpiSpace>> reqs;
    reqs.reserve(send_reqs_.size() + recv_reqs_.size());
    reqs.insert(reqs.end(), send_reqs_.begin(), send_reqs_.end());
    reqs.insert(reqs.end(), recv_reqs_.begin(), recv_reqs_.end());
    wait_all(reqs);
    Kokkos::Tools::popRegion();
  }

 private:
  std::vector<Req<MpiSpace>> send_reqs_;  // Queue for send requests
  std::vector<Req<MpiSpace>> recv_reqs_;  // Queue for receive requests
  int dest_rank_;                         // Destination rank for send
  int src_rank_;                          // Source rank for receive
  int tag_;                               // MPI tag
  MPI_Comm comm_;                         // MPI communicator
};

}  // namespace KokkosComm
