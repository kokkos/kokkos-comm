// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>
#if defined(KOKKOSCOMM_ENABLE_MPI)
#include <mpi.h>
#elif defined(KOKKOSCOMM_ENABLE_NCCL)
#include <nccl.h>
#endif

namespace {

template <typename T>
class DatatypeConversion : public testing::Test {
 public:
  using Datatype = T;
};

#if defined(KOKKOSCOMM_ENABLE_NCCL)
using DatatypeTypes = ::testing::Types<char, int, unsigned, std::int8_t, std::uint8_t, std::int32_t, std::uint32_t,
                                       std::int64_t, std::uint64_t, std::size_t, std::ptrdiff_t, float, double>;
#else
using DatatypeTypes =
    ::testing::Types<std::byte, char, unsigned char, short, unsigned short, int, unsigned, long, unsigned long,
                     long long, unsigned long long, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t,
                     std::int32_t, std::uint32_t, std::int64_t, std::uint64_t, std::size_t, std::ptrdiff_t, float,
                     double, long double, Kokkos::complex<float>, Kokkos::complex<double>>;
#endif
TYPED_TEST_SUITE(DatatypeConversion, DatatypeTypes);

template <KokkosComm::CommunicationSpace CS, typename T>
auto check_datatype_conversion(typename CS::datatype_type dtype) -> void {
#if defined(KOKKOSCOMM_ENABLE_MPI)
  if constexpr (std::is_same_v<T, std::byte>) {
    ASSERT_EQ(MPI_BYTE, dtype);
  } else if constexpr (std::is_same_v<T, char>) {
    ASSERT_EQ(MPI_CHAR, dtype);
  } else if constexpr (std::is_same_v<T, unsigned char>) {
    ASSERT_EQ(MPI_UNSIGNED_CHAR, dtype);
  } else if constexpr (std::is_same_v<T, short>) {
    ASSERT_EQ(MPI_SHORT, dtype);
  } else if constexpr (std::is_same_v<T, unsigned short>) {
    ASSERT_EQ(MPI_UNSIGNED_SHORT, dtype);
  } else if constexpr (std::is_same_v<T, int>) {
    ASSERT_EQ(MPI_INT, dtype);
  } else if constexpr (std::is_same_v<T, unsigned>) {
    ASSERT_EQ(MPI_UNSIGNED, dtype);
  } else if constexpr (std::is_same_v<T, long>) {
    ASSERT_EQ(MPI_LONG, dtype);
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    ASSERT_EQ(MPI_UNSIGNED_LONG, dtype);
  } else if constexpr (std::is_same_v<T, long long>) {
    ASSERT_EQ(MPI_LONG_LONG, dtype);
  } else if constexpr (std::is_same_v<T, unsigned long long>) {
    ASSERT_EQ(MPI_UNSIGNED_LONG_LONG, dtype);
  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    ASSERT_EQ(MPI_INT8_T, dtype);
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    ASSERT_EQ(MPI_UINT8_T, dtype);
  } else if constexpr (std::is_same_v<T, std::int16_t>) {
    ASSERT_EQ(MPI_INT16_T, dtype);
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    ASSERT_EQ(MPI_UINT16_T, dtype);
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    ASSERT_EQ(MPI_INT32_T, dtype);
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    ASSERT_EQ(MPI_UINT32_T, dtype);
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    ASSERT_EQ(MPI_INT64_T, dtype);
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    ASSERT_EQ(MPI_UINT64_T, dtype);
  } else if constexpr (std::is_same_v<T, std::size_t>) {
    if constexpr (sizeof(std::size_t) == 1) {
      ASSERT_EQ(MPI_UINT8_T, dtype);
    } else if constexpr (sizeof(std::size_t) == 2) {
      ASSERT_EQ(MPI_UINT16_T, dtype);
    } else if constexpr (sizeof(std::size_t) == 4) {
      ASSERT_EQ(MPI_UINT32_T, dtype);
    } else if constexpr (sizeof(std::size_t) == 8) {
      ASSERT_EQ(MPI_UINT64_T, dtype);
    }
  } else if constexpr (std::is_same_v<T, std::ptrdiff_t>) {
    if constexpr (sizeof(std::ptrdiff_t) == 1) {
      ASSERT_EQ(MPI_INT8_T, dtype);
    } else if constexpr (sizeof(std::ptrdiff_t) == 2) {
      ASSERT_EQ(MPI_INT16_T, dtype);
    } else if constexpr (sizeof(std::ptrdiff_t) == 4) {
      ASSERT_EQ(MPI_INT32_T, dtype);
    } else if constexpr (sizeof(std::ptrdiff_t) == 8) {
      ASSERT_EQ(MPI_INT64_T, dtype);
    }
  } else if constexpr (std::is_same_v<T, float>) {
    ASSERT_EQ(MPI_FLOAT, dtype);
  } else if constexpr (std::is_same_v<T, double>) {
    ASSERT_EQ(MPI_DOUBLE, dtype);
  } else if constexpr (std::is_same_v<T, long double>) {
    ASSERT_EQ(MPI_LONG_DOUBLE, dtype);
  } else if constexpr (std::is_same_v<T, Kokkos::complex<float>>) {
#if defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI)
    ASSERT_EQ(MPI_CXX_COMPLEX, dtype);
#else
    ASSERT_EQ(MPI_COMPLEX, dtype);
#endif
  } else if constexpr (std::is_same_v<T, Kokkos::complex<double>>) {
#if defined(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI)
    ASSERT_EQ(MPI_CXX_DOUBLE_COMPLEX, dtype);
#else
    ASSERT_EQ(MPI_DOUBLE_COMPLEX, dtype);
#endif
  }
#elif defined(KOKKOSCOMM_ENABLE_NCCL)
  if constexpr (std::is_same_v<T, char>) {
    ASSERT_EQ(ncclChar, dtype);
  } else if constexpr (std::is_same_v<T, int>) {
    ASSERT_EQ(ncclInt, dtype);
  } else if constexpr (std::is_same_v<T, unsigned> and sizeof(unsigned) == 4) {
    ASSERT_EQ(ncclUint32, dtype);
  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    ASSERT_EQ(ncclInt8, dtype);
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    ASSERT_EQ(ncclUint8, dtype);
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    ASSERT_EQ(ncclInt32, dtype);
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    ASSERT_EQ(ncclUint32, dtype);
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    ASSERT_EQ(ncclInt64, dtype);
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    ASSERT_EQ(ncclUint64, dtype);
  } else if constexpr (std::is_same_v<T, std::size_t>) {
    if constexpr (sizeof(std::size_t) == 1) {
      ASSERT_EQ(ncclUint8, dtype);
    } else if constexpr (sizeof(std::size_t) == 4) {
      ASSERT_EQ(ncclUint32, dtype);
    } else if constexpr (sizeof(std::size_t) == 8) {
      ASSERT_EQ(ncclUint64, dtype);
    }
  } else if constexpr (std::is_same_v<T, std::ptrdiff_t>) {
    if constexpr (sizeof(std::ptrdiff_t) == 1) {
      ASSERT_EQ(ncclInt8, dtype);
    } else if constexpr (sizeof(std::ptrdiff_t) == 4) {
      ASSERT_EQ(ncclInt32, dtype);
    } else if constexpr (sizeof(std::ptrdiff_t) == 8) {
      ASSERT_EQ(ncclInt64, dtype);
    }
  } else if constexpr (std::is_same_v<T, float>) {
    ASSERT_EQ(ncclFloat, dtype);
  } else if constexpr (std::is_same_v<T, double>) {
    ASSERT_EQ(ncclDouble, dtype);
  }
#else
  GTEST_SKIP() << "Unimplemented test for Communication Space";
#endif
}

using DCS = KokkosComm::DefaultCommunicationSpace;

TYPED_TEST(DatatypeConversion, DTypeConv) {
  using T    = typename TestFixture::Datatype;
  auto dtype = KokkosComm::datatype<DCS, T>();
  check_datatype_conversion<DCS, T>(dtype);
}

TYPED_TEST(DatatypeConversion, DTypeConvFromView) {
  using T = typename TestFixture::Datatype;
  Kokkos::View<T[10]> v("v");
  auto dtype = KokkosComm::datatype_for<DCS>(v);
  check_datatype_conversion<DCS, T>(dtype);
}

}  // namespace
