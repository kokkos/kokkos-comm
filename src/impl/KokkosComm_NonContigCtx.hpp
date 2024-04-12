#pragma once

#include <vector>
#include <memory>

#include "impl/KokkosComm_include_mpi.hpp"
#include "impl/KokkosComm_InvokableHolder.hpp"
#include "impl/KokkosComm_ViewHolder.hpp"

namespace KokkosComm {

struct Ctx {
  struct MpiArgs {
    void *buf;  // it seems like this could be void, but it's useful to have the
    // type information attached so that we can deep-copy to/from it
    // alternative is to create a matching unmanaged view type
    MPI_Datatype datatype;
    int count;
    MPI_Request req;

    MpiArgs() : buf(nullptr), datatype(MPI_DATATYPE_NULL), count(-1), req(MPI_REQUEST_NULL) {}
    MpiArgs(void *_buf, MPI_Datatype _datatype, int _count)
        : buf(_buf), datatype(_datatype), count(_count), req(MPI_REQUEST_NULL) {}
  };

  std::vector<MpiArgs> mpi_args;
  std::vector<std::shared_ptr<Impl::InvokableHolderBase>> wait_callbacks;
  std::vector<std::shared_ptr<Impl::ViewHolderBase>> views;  // views that need to stay alive as long as this thing
};
};  // namespace KokkosComm
