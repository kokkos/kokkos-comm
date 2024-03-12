#pragma once

#define KOKKOSCOMM_GCC_VERSION \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#if KOKKOSCOMM_GCC_VERSION >= 11400
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include <mpi.h>
#pragma GCC diagnostic pop
#else
#include <mpi.h>
#endif