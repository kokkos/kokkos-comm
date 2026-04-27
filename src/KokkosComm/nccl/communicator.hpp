// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <optional>

#include <cuda.h>
#include <nccl.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/fwd.hpp>
#include <KokkosComm/concepts.hpp>
#include "nccl_space.hpp"

namespace KokkosComm {

template <>
class Communicator<Experimental::NcclSpace, Kokkos::Cuda> {
 public:
  using execution_space     = Kokkos::Cuda;
  using communication_space = Experimental::NcclSpace;
  using communicator_type   = communication_space::communicator_type;
  using size_type           = communication_space::size_type;
  using rank_type           = communication_space::rank_type;

  /// @brief Constructs a `Communicator` from a raw `ncclComm_t` handle and a Kokkos CUDA execution space instance.
  /// Defaults `exec` to `Kokkos::Cuda`.
  ///
  /// The returned communicator does not own the underlying handle, and the user is responsible for destroying it.
  [[nodiscard]] static auto from_raw(communicator_type comm, const execution_space& exec = execution_space{}) noexcept
      -> Communicator<communication_space, execution_space> {
    return Communicator<communication_space, execution_space>(comm, exec, false);
  }

  /// @brief Splits from a raw NCCL communicator by color and key.
  ///
  /// Creates as many new communicators as distinct values of color are given, and orders processes according to the
  /// value of `key`. All processes with the same value of `color` join the same communicator.
  /// A process that passes `NCCL_SPLIT_NOCOLOR` as `color` will not join a new communicator and `nullopt` is returned.
  [[nodiscard]] static auto split_from_raw(
      const communicator_type comm, int color, int key, const execution_space& exec = execution_space{}
  ) noexcept -> std::optional<Communicator<communication_space, execution_space>> {
    communicator_type new_comm;
    ncclCommSplit(comm, color, key, &new_comm, nullptr);
    if (new_comm == nullptr) {
      return std::nullopt;
    }
    return Communicator<communication_space, execution_space>(new_comm, exec, true);
  }

  /// @brief Splits an existing communicator by color and key.
  [[nodiscard]] auto split(int color, int key) noexcept
      -> std::optional<Communicator<communication_space, execution_space>> {
    return Communicator::split_from_raw(comm_, color, key, exec_);
  }

  /// @brief Duplicates from a raw NCCL communicator.
  [[nodiscard]] static auto duplicate_from_raw(
      const communicator_type comm, const execution_space& exec = execution_space{}
  ) noexcept -> std::optional<Communicator<communication_space, execution_space>> {
    int rank;
    ncclCommUserRank(comm, &rank);
    return Communicator::split_from_raw(comm, 0, rank, exec);
  }

  /// @brief Duplicates an existing communicator.
  [[nodiscard]] auto duplicate() noexcept -> std::optional<Communicator<communication_space, execution_space>> {
    return Communicator::split_from_raw(comm_, 0, rank_, exec_);
  }

  /// @brief Destructor.
  ~Communicator() {
    if (owned_) {
      ncclCommDestroy(comm_);
    }
  }
  /// @brief Copy constructor is deleted because a `Communicator` cannot be implicitly copied.
  Communicator(const Communicator&) = delete;
  /// @brief Copy assignment operator is deleted because a `Communicator` cannot be implicitly copied.
  auto operator=(const Communicator&) -> Communicator& = delete;
  /// @brief Move constructor.
  Communicator(Communicator&& other) noexcept
      : comm_(std::exchange(other.comm_, nullptr)),
        exec_(std::move(other.exec_)),
        size_(std::exchange(other.size_, 0)),
        rank_(std::exchange(other.rank_, 0)),
        owned_(std::exchange(other.owned_, false)) {}
  /// @brief Move assignment operator.
  auto operator=(Communicator&& other) noexcept -> Communicator& {
    // Self-assignment guard is necessary here since the move assignment operator can be called on an already
    // initialized object, and we must prevent self-move from calling `MPI_Comm_free`.
    if (this != &other) {
      // Run destructor logic on current state before overwriting (if `this` is already initialized)
      if (owned_) {
        ncclCommDestroy(comm_);
      }
      comm_  = std::exchange(other.comm_, nullptr);
      exec_  = std::move(other.exec_);
      size_  = std::exchange(other.size_, 0);
      rank_  = std::exchange(other.rank_, 0);
      owned_ = std::exchange(other.owned_, false);
    }
    return *this;
  }

  [[nodiscard]] constexpr auto exec() const noexcept -> const execution_space& { return exec_; }
  [[nodiscard]] constexpr auto comm() const noexcept -> const communicator_type& { return comm_; }
  // Non-const overload required because most MPI communication primitives take `ncclComm_t` by value
  [[nodiscard]] constexpr auto comm() noexcept -> communicator_type& { return comm_; }
  [[nodiscard]] constexpr auto size() const noexcept -> size_type { return size_; }
  [[nodiscard]] constexpr auto rank() const noexcept -> rank_type { return rank_; }

 private:
  explicit Communicator(communicator_type comm, const execution_space& exec, bool owned)
      : exec_(exec), comm_(comm), size_(0), rank_(0), owned_(owned) {
    set_size();
    set_rank();
  }

  auto set_size() noexcept -> void { ncclCommCount(comm_, &size_); }
  auto set_rank() noexcept -> void { ncclCommUserRank(comm_, &rank_); }

  execution_space exec_;
  communicator_type comm_;
  size_type size_;
  rank_type rank_;
  bool owned_;
};

}  // namespace KokkosComm
