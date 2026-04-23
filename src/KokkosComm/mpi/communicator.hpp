// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <optional>

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/fwd.hpp>
#include <KokkosComm/concepts.hpp>
#include "mpi_space.hpp"

namespace KokkosComm {

template <KokkosExecutionSpace Exec>
class Communicator<MpiSpace, Exec> {
 public:
  using execution_space     = Exec;
  using communication_space = MpiSpace;
  using communicator_type   = communication_space::communicator_type;
  using size_type           = communication_space::size_type;
  using rank_type           = communication_space::rank_type;

  /// @brief Constructs a `Communicator` from a raw `MPI_Comm` handle and a Kokkos execution space instance.
  /// Defaults `exec` to `Kokkos::DefaultExecutionSpace`.
  ///
  /// The passed `comm` must be a valid handle and must not be an inter-communicator parent handle.
  /// The returned communicator does not own the underlying handle, and the user is responsible for destroying it.
  [[nodiscard]] static auto from_raw(communicator_type comm, const execution_space& exec = execution_space{}) noexcept
      -> Communicator<communication_space, execution_space> {
    return Communicator(comm, exec, false);
  }

  /// @brief Splits from a raw MPI communicator handle by color and key.
  ///
  /// Creates as many new communicators as distinct values of color are given, and orders processes according to the
  /// value of `key`. All processes with the same value of `color` join the same communicator.
  /// A process that passes `MPI_UNDEFINED` as `color` will not join a new communicator and `nullopt` is returned.
  [[nodiscard]] static auto split_from_raw(
      const communicator_type comm, int color, int key, const execution_space& exec = execution_space{}
  ) noexcept -> std::optional<Communicator<communication_space, execution_space>> {
    communicator_type new_comm;
    MPI_Comm_split(comm, color, key, &new_comm);
    // Something may have failed, but `color` may be `MPI_UNDEFINED` and we are now outside the split communicator
    if (new_comm == MPI_COMM_NULL) {
      return std::nullopt;
    }
    return Communicator<communication_space, execution_space>(new_comm, exec, true);
  }

  /// @brief Splits an existing communicator by color and key.
  [[nodiscard]] auto split(int color, int key) -> std::optional<Communicator<communication_space, execution_space>> {
    return Communicator::split_from_raw(comm_, color, key, exec_);
  }

  /// @brief Duplicates from a raw MPI communicator.
  ///
  /// If `MPI_Comm_dup` fails, returns `nullopt`.
  [[nodiscard]] static auto duplicate_from_raw(
      const communicator_type comm, const execution_space& exec = execution_space{}
  ) noexcept -> std::optional<Communicator<communication_space, execution_space>> {
    communicator_type new_comm;
    MPI_Comm_dup(comm, &new_comm);
    // Something failed, but we don't have proper error handling yet :(
    if (new_comm == MPI_COMM_NULL) {
      return std::nullopt;
    }
    return Communicator<communication_space, execution_space>(new_comm, exec, true);
  }

  /// @brief Duplicates an existing communicator.
  [[nodiscard]] auto duplicate() -> std::optional<Communicator<communication_space, execution_space>> {
    return Communicator::duplicate_from_raw(comm_, exec_);
  }

  /// @brief Destructor.
  ~Communicator() noexcept {
    // Free the underlying `MPI_Comm` if it is owned, and is not `MPI_COMM_WORLD`, nor `MPI_COMM_SELF`.
    // NOTE: `MPI_COMM_WORLD` and `MPI_COMM_SELF` are predefined communicators that shall not be passed to
    // `MPI_Comm_free` from user code.
    if (owned_ && comm_ != MPI_COMM_WORLD && comm_ != MPI_COMM_SELF) {
      MPI_Comm_free(&comm_);
    }
  }
  /// @brief Copy constructor is deleted because a `Communicator` cannot be implicitly copied.
  /// Use `duplicate` instead.
  Communicator(const Communicator&) = delete;
  /// @brief Copy assignment operator is deleted because a `Communicator` cannot be implicitly copied.
  /// Use `duplicate` instead.
  auto operator=(const Communicator&) -> Communicator& = delete;
  /// @brief Move constructor.
  Communicator(Communicator&& other) noexcept
      : comm_(std::exchange(other.comm_, MPI_COMM_NULL)),
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
      if (owned_ && comm_ != MPI_COMM_WORLD && comm_ != MPI_COMM_SELF) {
        MPI_Comm_free(&comm_);
      }
      comm_  = std::exchange(other.comm_, MPI_COMM_NULL);
      exec_  = std::move(other.exec_);
      size_  = std::exchange(other.size_, 0);
      rank_  = std::exchange(other.rank_, 0);
      owned_ = std::exchange(other.owned_, false);
    }
    return *this;
  }

  [[nodiscard]] constexpr auto exec() const noexcept -> const execution_space& { return exec_; }
  [[nodiscard]] constexpr auto comm() const noexcept -> const communicator_type& { return comm_; }
  // Non-const overload required because most MPI communication primitives take `MPI_Comm` by value
  [[nodiscard]] constexpr auto comm() noexcept -> communicator_type& { return comm_; }
  [[nodiscard]] constexpr auto size() const noexcept -> size_type { return size_; }
  [[nodiscard]] constexpr auto rank() const noexcept -> rank_type { return rank_; }

 private:
  explicit Communicator(communicator_type comm, const execution_space& exec, bool owned)
      : comm_(comm), exec_(exec), size_(0), rank_(0), owned_(owned) {
    set_size();
    set_rank();
  }

  auto set_size() noexcept -> void { MPI_Comm_size(comm_, &size_); }
  auto set_rank() noexcept -> void { MPI_Comm_rank(comm_, &rank_); }

  communicator_type comm_;
  execution_space exec_;
  size_type size_;
  rank_type rank_;
  bool owned_;
};

}  // namespace KokkosComm
