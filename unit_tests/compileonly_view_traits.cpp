#include <KokkosComm.hpp>

struct FakeView {};

namespace KokkosComm {

// make the FakeView implement the view interface
template <>
struct Traits<FakeView> {
  using pointer_type = char *;
  using scalar_type  = char;

  static bool is_contiguous(const FakeView &v) { return false; }

  static pointer_type data_handle(const FakeView &v) { return nullptr; }

  static size_t span(const FakeView &v) { return 0; }

  static size_t extent(const FakeView &v, const int i) { return 0; }

  static size_t stride(const FakeView &v, const int i) { return 0; }

  static constexpr bool is_reference_counted() { return false; }

  static constexpr size_t rank() { return 0; }
};

// Make the FakeView implement the pack interface
template <>
struct PackTraits<FakeView> {
  using packer_type = Impl::Packer::MpiDatatype<FakeView>;

  static bool needs_unpack(const FakeView &v) { return false; }
  static bool needs_pack(const FakeView &v) { return false; }
};

}  // namespace KokkosComm

// should compile
int main() {
  KokkosComm::isend(Kokkos::DefaultExecutionSpace(), FakeView(), 0, 0,
                    MPI_COMM_WORLD);
}