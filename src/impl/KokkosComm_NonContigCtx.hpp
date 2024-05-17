#pragma once

#include <vector>
#include <memory>

#include "impl/KokkosComm_include_mpi.hpp"
#include "impl/KokkosComm_InvokableHolder.hpp"
#include "impl/KokkosComm_ViewHolder.hpp"

namespace KokkosComm {

class CtxBase {
 public:
  CtxBase() : pre_uses_space_(false), post_uses_space_(false) {}

  bool pre_uses_space() const { return pre_uses_space_; }
  void set_pre_uses_space(bool val = true) { pre_uses_space_ = val; }
  bool post_uses_space() const { return post_uses_space_; }
  void set_post_uses_space(bool val = true) { post_uses_space_ = val; }

  std::vector<std::shared_ptr<Impl::InvokableHolderBase>> wait_callbacks;
  std::vector<std::shared_ptr<Impl::ViewHolderBase>> views;  // views that need to stay alive as long as this thing

 protected:
  bool pre_uses_space_;
  bool post_uses_space_;
};

struct CtxBufCount : public CtxBase {
  struct MpiArgs {
    void *buf;
    MPI_Datatype datatype;
    MPI_Request req;
    int count;

    MpiArgs() : buf(nullptr), datatype(MPI_DATATYPE_NULL), req(MPI_REQUEST_NULL), count(-1) {}
    MpiArgs(void *_buf, MPI_Datatype _datatype, int _count)
        : buf(_buf), datatype(_datatype), req(MPI_REQUEST_NULL), count(_count) {}
  };

  CtxBufCount() = default;

  std::vector<MpiArgs> mpi_args;
};

struct CtxReduce : public CtxBase {
  struct MpiArgs {
    void *sbuf;
    void *rbuf;
    MPI_Datatype datatype;
    int count;
    MPI_Request req;

    MpiArgs(void *_sbuf, void *_rbuf, MPI_Datatype _datatype, int _count)
        : sbuf(_sbuf), rbuf(_rbuf), datatype(_datatype), count(_count), req(MPI_REQUEST_NULL) {}
  };

  CtxReduce() = default;

  std::vector<MpiArgs> mpi_args;
};

struct CtxAlltoall : public CtxBase {
  struct MpiArgs {
    void *sbuf;
    void *rbuf;
    int scount;
    int rcount;
    MPI_Datatype stype;
    MPI_Datatype rtype;
    MPI_Request req;

    MpiArgs(void *_sbuf, int _scount, MPI_Datatype _stype, void *_rbuf, int _rcount, MPI_Datatype _rtype)
        : sbuf(_sbuf),
          scount(_scount),
          stype(_stype),
          rbuf(_rbuf),
          rcount(_rcount),
          rtype(_rtype),
          req(MPI_REQUEST_NULL) {}
  };

  CtxAlltoall() = default;

  std::vector<MpiArgs> mpi_args;
};

};  // namespace KokkosComm
