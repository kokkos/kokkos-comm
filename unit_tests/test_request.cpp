// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/KokkosComm.hpp>

using RequestType = KokkosComm::Request<KokkosComm::DefaultCommunicationSpace>;

static_assert(!std::is_copy_constructible_v<RequestType>);
static_assert(!std::is_copy_assignable_v<RequestType>);
static_assert(std::is_nothrow_move_constructible_v<RequestType>);
static_assert(std::is_nothrow_move_assignable_v<RequestType>);

auto make_ready_request(RequestType& request) -> void {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  Kokkos::Cuda space;
  request.capture_stream_state(space.cuda_stream());
  EXPECT_EQ(cudaStreamSynchronize(space.cuda_stream()), cudaSuccess);
#else
  EXPECT_EQ(MPI_Ibarrier(MPI_COMM_SELF, request.request_ptr()), MPI_SUCCESS);
#endif
}

// A receive without a matching send, or an event behind a gated CUDA host function.
// The same batch tests exercise pending and ready operations on either backend.
class PendingRequestOperation {
 public:
  explicit PendingRequestOperation(RequestType& request) : request_(request) {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
    EXPECT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
    EXPECT_EQ(
        cudaLaunchHostFunc(
            stream_,
            [](void* operation) {
              auto& pending = *static_cast<PendingRequestOperation*>(operation);
              while (!pending.release_.load(std::memory_order_acquire)) {
              }
              pending.finished_.store(true, std::memory_order_release);
            },
            this
        ),
        cudaSuccess
    );
    request_.capture_stream_state(stream_);
#else
    EXPECT_EQ(MPI_Irecv(&received_, 1, MPI_INT, 0, 126, MPI_COMM_SELF, request_.request_ptr()), MPI_SUCCESS);
#endif
  }

  auto complete() -> void {
    if (completed_) return;
#if defined(KOKKOSCOMM_ENABLE_NCCL)
    release_.store(true, std::memory_order_release);
#else
    MPI_Request send = MPI_REQUEST_NULL;
    EXPECT_EQ(MPI_Isend(&sent_, 1, MPI_INT, 0, 126, MPI_COMM_SELF, &send), MPI_SUCCESS);
    EXPECT_EQ(MPI_Wait(&send, MPI_STATUS_IGNORE), MPI_SUCCESS);
#endif
    completed_ = true;
  }

  auto expect_completed() const -> void {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
    EXPECT_TRUE(finished_.load(std::memory_order_acquire));
#else
    EXPECT_EQ(received_, sent_);
#endif
  }

  ~PendingRequestOperation() {
    complete();
    request_.wait();
#if defined(KOKKOSCOMM_ENABLE_NCCL)
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
#endif
    expect_completed();
  }

 private:
  RequestType& request_;
  bool completed_ = false;
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  cudaStream_t stream_ = nullptr;
  std::atomic<bool> release_{false};
  std::atomic<bool> finished_{false};
#else
  int received_ = -1;
  int sent_     = 42;
#endif
};

TEST(Request, BatchWaitsHandleEmptyAndInactiveRequests) {
  std::span<RequestType> empty;
  KokkosComm::wait_all(empty);
  EXPECT_EQ(KokkosComm::wait_any(empty), std::nullopt);

  std::array<RequestType, 2> requests;
  int callbacks = 0;
  requests[1].add_callback([&] { ++callbacks; });
  EXPECT_EQ(KokkosComm::wait_any(requests), std::nullopt);
  EXPECT_EQ(callbacks, 0);  // No active handle was selected.
  KokkosComm::wait_all(requests);
  EXPECT_EQ(callbacks, 1);
  KokkosComm::wait_all(requests);
  EXPECT_EQ(KokkosComm::wait_any(requests), std::nullopt);
  EXPECT_EQ(callbacks, 1);
}

TEST(Request, WaitAllRetiresEveryRequestBeforeCallbacks) {
  std::array<RequestType, 4> requests;
  std::array<int, 4> callbacks{};
  Kokkos::View<int*, Kokkos::HostSpace> view("batch_retained", 1);
  const auto initial_count = view.use_count();
  const auto null_handle   = requests[0].request();
  make_ready_request(requests[1]);
  PendingRequestOperation pending(requests[3]);
  for (size_t r = 0; r < requests.size(); ++r) {
    requests[r].extend_view_lifetime(view);
    requests[r].add_callback([&, r] {
      pending.expect_completed();
      for (const auto& request : requests) EXPECT_EQ(request.request(), null_handle);
      EXPECT_GT(view.use_count(), initial_count);
      ++callbacks[r];
    });
  }
  // Delay one operation to exercise polling while another is already ready.
  std::thread progress([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pending.complete();
  });
  KokkosComm::wait_all(requests);
  progress.join();
  EXPECT_EQ(callbacks, (std::array<int, 4>{1, 1, 1, 1}));
  EXPECT_EQ(view.use_count(), initial_count);
  KokkosComm::wait_all(requests);
  EXPECT_EQ(KokkosComm::wait_any(requests), std::nullopt);
  EXPECT_EQ(callbacks, (std::array<int, 4>{1, 1, 1, 1}));
  make_ready_request(requests[1]);
  requests[1].add_callback([&] { ++callbacks[1]; });
  KokkosComm::wait_all(requests);
  EXPECT_EQ(callbacks, (std::array<int, 4>{1, 2, 1, 1}));
}

TEST(Request, WaitAnyRetiresOnlySelectedRequestAndRestoresOthers) {
  std::array<RequestType, 4> requests;
  std::array<int, 4> callbacks{};
  Kokkos::View<int*, Kokkos::HostSpace> view("batch_any_retained", 1);
  const auto initial_count = view.use_count();
  const auto null_handle   = requests[0].request();
  PendingRequestOperation pending(requests[1]);
  const auto pending_handle = requests[1].request();
  make_ready_request(requests[3]);
  for (size_t r : {0u, 1u, 3u}) {
    requests[r].extend_view_lifetime(view);
    requests[r].add_callback([&, r] {
      EXPECT_EQ(requests[r].request(), null_handle);
      EXPECT_GT(view.use_count(), initial_count);
      if (r == 3) EXPECT_EQ(requests[1].request(), pending_handle);
      if (r == 1) pending.expect_completed();
      ++callbacks[r];
    });
  }
  EXPECT_EQ(KokkosComm::wait_any(requests), 3);
  EXPECT_EQ(requests[1].request(), pending_handle);
  EXPECT_EQ(callbacks, (std::array<int, 4>{0, 0, 0, 1}));
  EXPECT_EQ(view.use_count(), initial_count + 2);
  std::thread progress([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pending.complete();
  });
  EXPECT_EQ(KokkosComm::wait_any(requests), 1);
  progress.join();
  EXPECT_EQ(view.use_count(), initial_count + 1);
  EXPECT_EQ(KokkosComm::wait_any(requests), std::nullopt);
  EXPECT_EQ(callbacks, (std::array<int, 4>{0, 1, 0, 1}));
  KokkosComm::wait_all(requests);  // Drain the callback-only inactive request.
  EXPECT_EQ(callbacks, (std::array<int, 4>{1, 1, 0, 1}));
  EXPECT_EQ(view.use_count(), initial_count);
  make_ready_request(requests[3]);
  requests[3].add_callback([&] { ++callbacks[3]; });
  EXPECT_EQ(KokkosComm::wait_any(requests), 3);
  EXPECT_EQ(KokkosComm::wait_any(requests), std::nullopt);
  EXPECT_EQ(callbacks[3], 2);
}

TEST(Request, WaitAnySelectsEachReadyRequestExactlyOnce) {
  std::array<RequestType, 4> requests;
  std::array<int, 4> callbacks{};
  std::array<RequestType::request_type, 4> handles{};
  const auto null_handle = requests[0].request();
  for (size_t r : {1u, 3u}) {
    make_ready_request(requests[r]);
    handles[r] = requests[r].request();
    requests[r].add_callback([&, r] {
      const auto other = r == 1 ? 3 : 1;
      EXPECT_EQ(requests[r].request(), null_handle);
      EXPECT_EQ(requests[other].request(), callbacks[other] == 0 ? handles[other] : null_handle);
      ++callbacks[r];
    });
  }
  const auto first = KokkosComm::wait_any(requests);
  ASSERT_TRUE(first == 1 || first == 3);
  const auto other = *first == 1 ? 3 : 1;
  EXPECT_EQ(requests[*first].request(), null_handle);
  EXPECT_NE(requests[other].request(), null_handle);
  EXPECT_EQ(callbacks[*first], 1);
  EXPECT_EQ(callbacks[other], 0);
  EXPECT_EQ(KokkosComm::wait_any(requests), other);
  EXPECT_EQ(requests[other].request(), null_handle);
  EXPECT_EQ(KokkosComm::wait_any(requests), std::nullopt);
  EXPECT_EQ(callbacks, (std::array<int, 4>{0, 1, 0, 1}));
}

template <class Request, class View, class MakeProgress>
auto expect_completion_through_test(Request& request, const View& view, MakeProgress&& make_progress) -> void {
  const auto initial_use_count = view.use_count();
  std::vector<int> callback_order;
  request.extend_view_lifetime(view);
  EXPECT_GT(view.use_count(), initial_use_count);
  request.add_callback([&] {
    EXPECT_GT(view.use_count(), initial_use_count);
    callback_order.push_back(1);
  });
  request.add_callback([&] {
    EXPECT_GT(view.use_count(), initial_use_count);
    callback_order.push_back(2);
  });

  EXPECT_FALSE(request.test());
  EXPECT_TRUE(callback_order.empty());
  std::forward<MakeProgress>(make_progress)();
  while (!request.test()) {
  }

  EXPECT_EQ(callback_order, (std::vector<int>{1, 2}));
  EXPECT_EQ(view.use_count(), initial_use_count);
  EXPECT_TRUE(request.test());
  request.wait();
  EXPECT_EQ(callback_order, (std::vector<int>{1, 2}));
  EXPECT_EQ(view.use_count(), initial_use_count);
}

TEST(Request, EmptyCompletionIsIdempotent) {
  RequestType request;
  EXPECT_TRUE(request.test());
  request.wait();
  KokkosComm::wait(request);
  EXPECT_TRUE(KokkosComm::test(request));
  KokkosComm::wait(std::move(request));
}

TEST(Request, MovesTransferCallbacksAndManagedAllocations) {
  Kokkos::View<int*, Kokkos::HostSpace> view("retained", 1);
  const auto initial_count = view.use_count();
  std::vector<int> order;
  RequestType destination;
  {
    RequestType source;
    source.extend_view_lifetime(view);
    source.add_callback([&] { order.push_back(1); });
    source.add_callback([&] {
      EXPECT_GT(view.use_count(), initial_count);
      order.push_back(2);
    });
    EXPECT_GT(view.use_count(), initial_count);
    RequestType moved(std::move(source));
    EXPECT_TRUE(source.test());
    source.wait();
    destination = std::move(moved);
    EXPECT_TRUE(moved.test());
  }
  EXPECT_TRUE(order.empty());
  EXPECT_GT(view.use_count(), initial_count);
  auto& alias = destination;
  destination = std::move(alias);
  EXPECT_TRUE(destination.test());
  EXPECT_EQ(order, (std::vector<int>{1, 2}));
  EXPECT_EQ(view.use_count(), initial_count);
  destination.wait();
  EXPECT_TRUE(destination.test());
  EXPECT_EQ(order.size(), 2u);
  RequestType next;
  next.add_callback([&] { order.push_back(3); });
  destination = std::move(next);
  destination.wait();
  EXPECT_EQ(order, (std::vector<int>{1, 2, 3}));
}

TEST(Request, UnmanagedAllocationsRemainCallerOwned) {
  int storage = 42;
  Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> view(&storage, 1);
  RequestType request;
  request.extend_view_lifetime(view);
  EXPECT_TRUE(request.test());
  EXPECT_EQ(view.use_count(), 0);
  EXPECT_EQ(storage, 42);
}

#if defined(KOKKOSCOMM_ENABLE_NCCL)

TEST(Request, ActiveMovesAndVectorRelocationPreserveCompletion) {
  Kokkos::Cuda space;
  Kokkos::View<int*, Kokkos::CudaSpace> view("request_data", 1);
  Kokkos::deep_copy(space, view, 42);
  RequestType destination;
  RequestType::request_type native;
  int callbacks = 0;
  // Registration before event capture is used by the receive/reduce implementations.
  {
    RequestType source;
    source.extend_view_lifetime(view);
    source.add_callback([&] { ++callbacks; });
    source.capture_stream_state(space.cuda_stream());
    native = source.request();
    RequestType moved(std::move(source));
    EXPECT_EQ(source.request(), nullptr);
    EXPECT_TRUE(source.test());
    EXPECT_EQ(moved.request(), native);
    std::vector<RequestType> requests;
    requests.reserve(1);
    requests.push_back(std::move(moved));
    requests.emplace_back();
    destination = std::move(requests.front());
    EXPECT_EQ(requests.front().request(), nullptr);
  }
  auto& alias = destination;
  destination = std::move(alias);
  EXPECT_EQ(destination.request(), native);
  destination.wait();
  EXPECT_EQ(destination.request(), nullptr);
  EXPECT_EQ(callbacks, 1);
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  EXPECT_EQ(host(0), 42);
  EXPECT_TRUE(destination.test());
  EXPECT_EQ(callbacks, 1);
  destination.capture_stream_state(space.cuda_stream());
  while (!destination.test()) {
  }
  EXPECT_EQ(destination.request(), nullptr);
}

TEST(Request, IncompleteTestPreservesEventAndCallbacks) {
  Kokkos::Cuda space;
  std::atomic<bool> release{false};
  ASSERT_EQ(
      cudaLaunchHostFunc(
          space.cuda_stream(),
          [](void* flag) {
            auto& ready = *static_cast<std::atomic<bool>*>(flag);
            while (!ready.load(std::memory_order_acquire)) {
            }
          },
          &release
      ),
      cudaSuccess
  );
  RequestType request;
  int callbacks = 0;
  request.add_callback([&] { ++callbacks; });
  request.capture_stream_state(space.cuda_stream());
  auto event = request.request();
  EXPECT_FALSE(request.test());
  EXPECT_EQ(request.request(), event);
  EXPECT_EQ(callbacks, 0);
  release.store(true, std::memory_order_release);
  request.wait();
  EXPECT_EQ(request.request(), nullptr);
  EXPECT_EQ(callbacks, 1);
}

TEST(Request, SuccessfulTestRetiresCallbacksAndManagedView) {
  Kokkos::Cuda space;
  Kokkos::View<int*, Kokkos::CudaSpace> view("poll_data", 1);
  std::atomic<bool> release{false};
  ASSERT_EQ(
      cudaLaunchHostFunc(
          space.cuda_stream(),
          [](void* flag) {
            auto& ready = *static_cast<std::atomic<bool>*>(flag);
            while (!ready.load(std::memory_order_acquire)) {
            }
          },
          &release
      ),
      cudaSuccess
  );
  Kokkos::deep_copy(space, view, 42);
  RequestType request;
  request.capture_stream_state(space.cuda_stream());
  expect_completion_through_test(request, view, [&] { release.store(true, std::memory_order_release); });
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  EXPECT_EQ(host(0), 42);
}

#else

static_assert(!std::is_constructible_v<RequestType, MPI_Request&>);
static_assert(std::is_nothrow_constructible_v<RequestType, MPI_Request&&>);

TEST(Request, ActiveMovesAndVectorRelocationPreserveCompletion) {
  int received = -1;
  int sent = 42;
  MPI_Request raw = MPI_REQUEST_NULL;
  ASSERT_EQ(MPI_Irecv(&received, 1, MPI_INT, 0, 123, MPI_COMM_SELF, &raw), MPI_SUCCESS);
  RequestType destination;
  RequestType::request_type native;
  int callbacks = 0;
  // Regression scenario from PR #282: destroy the source before completing the destination.
  // https://github.com/kokkos/kokkos-comm/pull/282
  {
    RequestType source(std::move(raw));
    EXPECT_EQ(raw, MPI_REQUEST_NULL);
    source.add_callback([&] { ++callbacks; });
    EXPECT_FALSE(source.test());  // No matching send has been posted.
    EXPECT_EQ(callbacks, 0);
    native = source.request();  // Borrowed identity comparison only.
    RequestType moved(std::move(source));
    EXPECT_EQ(source.request(), MPI_REQUEST_NULL);
    EXPECT_EQ(moved.request(), native);
    EXPECT_TRUE(source.test());
    std::vector<RequestType> requests;
    requests.reserve(1);
    requests.push_back(std::move(moved));
    requests.emplace_back();  // Forces relocation of the active receive.
    EXPECT_EQ(moved.request(), MPI_REQUEST_NULL);
    destination = std::move(requests.front());
    EXPECT_EQ(requests.front().request(), MPI_REQUEST_NULL);
  }
  auto& alias = destination;
  destination = std::move(alias);
  EXPECT_EQ(destination.request(), native);
  RequestType send;
  ASSERT_EQ(MPI_Isend(&sent, 1, MPI_INT, 0, 123, MPI_COMM_SELF, send.request_ptr()), MPI_SUCCESS);
  send.wait();
  destination.wait();
  EXPECT_EQ(destination.request(), MPI_REQUEST_NULL);
  EXPECT_EQ(received, sent);
  EXPECT_EQ(callbacks, 1);
  EXPECT_TRUE(destination.test());
  destination.wait();
  EXPECT_EQ(callbacks, 1);
}

TEST(Request, SuccessfulTestRetiresCallbacksAndManagedView) {
  Kokkos::View<int*, Kokkos::HostSpace> received("poll_recv", 1);
  RequestType request;
  ASSERT_EQ(MPI_Irecv(received.data(), 1, MPI_INT, 0, 124, MPI_COMM_SELF, request.request_ptr()), MPI_SUCCESS);
  int sent = 17;
  expect_completion_through_test(request, received, [&] {
    RequestType send;
    EXPECT_EQ(MPI_Isend(&sent, 1, MPI_INT, 0, 124, MPI_COMM_SELF, send.request_ptr()), MPI_SUCCESS);
    send.wait();
  });
  EXPECT_EQ(received(0), sent);
}

#endif
