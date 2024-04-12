#pragma once

namespace KokkosComm::Impl {
enum class CommMode {
  // Standard mode: MPI implementation decides whether outgoing messages will
  // be buffered. Send operations can be started whether or not a matching
  // receive has been started. They may complete before a matching receive is
  // started. Standard mode is non-local: successful completion of the send
  // operation may depend on the occurrence of a matching receive.
  Standard,
  // Ready mode: Send operations may be started only if the matching receive is
  // already started.
  Ready,
  // Synchronous mode: Send operations complete successfully only if a matching
  // receive is started, and the receive operation has started to receive the
  // message sent.
  Sync,
  Async,
};

template <CommMode mode>
struct send_return {
  using type = void;
};

template <>
struct send_return<CommMode::Async> {
  using type = Impl::Req;
};

template <CommMode mode>
using send_return_t = typename send_return<mode>::type;

}  // namespace KokkosComm::Impl