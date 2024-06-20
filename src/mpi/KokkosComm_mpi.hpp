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

#pragma once

#include <type_traits>

#include "KokkosComm_api.hpp"
#include "KokkosComm_concepts.hpp"
#include "KokkosComm_include_mpi.hpp"

namespace KokkosComm {

struct Mpi {
  /*
  - init_fence
  - allocations
  - pre_copies
  - pre_comm_fence
  - comm
  */
  template <KokkosExecutionSpace ExecSpace>
  class Handle {
   public:
    using execution_space = ExecSpace;

    // a type-erased view. Request uses these to keep temporary views alive for
    // the lifetime of "Immediate" MPI operations
    struct ViewHolderBase {
      virtual ~ViewHolderBase() {}
    };
    template <typename V>
    struct ViewHolder : ViewHolderBase {
      ViewHolder(const V &v) : v_(v) {}
      V v_;
    };

    Handle(const execution_space &space, MPI_Comm comm) : space_(space), comm_(comm), preCommFence_(false) {}

    // template <KokkosView View>
    // void impl_track_view(const View &v) {
    //     views_.push_back(std::make_shared<ViewHolder<View>>(v));
    // }

    void impl_add_pre_comm_fence() { preCommFence_ = true; }

    void impl_track_mpi_request(MPI_Request req) { reqs_.push_back(req); }

    void impl_add_alloc(std::function<void()> f) { allocs_.push_back(f); }

    void impl_add_pre_copy(std::function<void()> f) { preCopies_.push_back(f); }

    void impl_add_comm(std::function<void()> f) { comms_.push_back(f); }

    void impl_add_post_wait(std::function<void()> f) { postWaits_.push_back(f); }

    MPI_Comm &mpi_comm() { return comm_; }

    const execution_space &space() const { return space_; }

    void impl_run() {
      for (const auto &f : allocs_) f();
      for (const auto &f : preCopies_) f();
      if (preCommFence_) {
        space_.fence("pre-comm fence");
      }
      for (const auto &f : comms_) f();

      allocs_.clear();
      preCopies_.clear();
      comms_.clear();
    }

    void wait() {
      MPI_Waitall(reqs_.size(), reqs_.data(), MPI_STATUSES_IGNORE);
      reqs_.clear();
      // std::cerr << __FILE__ << ":" << __LINE__ << " MPI_Waitall done\n";
      for (const auto &f : postWaits_) {
        f();
      }
      // std::cerr << __FILE__ << ":" << __LINE__ << " postWaits_.clear()...\n";
      postWaits_.clear();
      // views_.clear();
      // std::cerr << __FILE__ << ":" << __LINE__ << " wait() done\n";
    }

   private:
    execution_space space_;
    MPI_Comm comm_;

    // phase variables
    bool preCommFence_;
    std::vector<std::function<void()>> allocs_;
    std::vector<std::function<void()>> preCopies_;
    std::vector<std::function<void()>> comms_;

    // wait variables
    std::vector<MPI_Request> reqs_;
    std::vector<std::function<void()>> postWaits_;
    std::vector<std::shared_ptr<ViewHolderBase>> views_;
  };

  template <KokkosExecutionSpace ExecSpace, typename ExchFunc>
  class Plan {
   public:
    using execution_space = ExecSpace;
    using handle_type     = Handle<execution_space>;
    Plan(const execution_space &space, MPI_Comm comm, ExchFunc f) : handle_(space, comm) {
      f(handle_);
      handle_.impl_run();
    }

    Plan(const execution_space &space, ExchFunc f) : Plan(space, MPI_COMM_WORLD, f) {}
    Plan(MPI_Comm comm, ExchFunc f) : Plan(Kokkos::DefaultExecutionSpace(), comm, f) {}
    Plan(ExchFunc f) : Plan(Kokkos::DefaultExecutionSpace(), MPI_COMM_WORLD, f) {}

    handle_type handle() const { return handle_; }

   private:
    handle_type handle_;
  };

  // add an isend to the plan
  template <KokkosExecutionSpace ExecSpace, KokkosView SendView>
  static void isend(Handle<ExecSpace> &handle, const SendView &sv, int dest, int tag);

  // add an irecv to the plan
  template <KokkosExecutionSpace ExecSpace, KokkosView Recv>
  static void irecv(Handle<ExecSpace> &handle, const Recv &rv, int src, int tag);

};  // struct Mpi

namespace Impl {
// catch-all, mark APIs as unimplemented
template <Impl::Api API>
struct api_avail<Mpi, API> : public std::false_type {};

// implemented APIs
template <>
struct api_avail<Mpi, Impl::Api::Irecv> : public std::true_type {};
template <>
struct api_avail<Mpi, Impl::Api::Isend> : public std::true_type {};
};  // namespace Impl

}  // namespace KokkosComm

#include "KokkosComm_mpi_irecv.hpp"
#include "KokkosComm_mpi_isend.hpp"