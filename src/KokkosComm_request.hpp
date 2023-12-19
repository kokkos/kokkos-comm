// Copyright 2023 Carl Pearson
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>

#include <mpi.h>

namespace KokkosComm {

class Req {

  // a type-erased view. Request uses these to keep temporary views alive for
  // the lifetime of "Immediate" MPI operations
  struct ViewHolderBase {
    virtual ~ViewHolderBase() {}
  };
  template <typename V> struct ViewHolder : ViewHolderBase {
    ViewHolder(const V &v) : v_(v) {}
    V v_;
  };

  struct Record {
    Record() : req_(MPI_REQUEST_NULL) {}
    MPI_Request req_;
    std::vector<std::shared_ptr<ViewHolderBase>> until_waits_;
  };

public:
  Req() : record_(std::make_shared<Record>()) {}

  MPI_Request &mpi_req() { return record_->req_; }

  void wait() {
    MPI_Wait(&(record_->req_), MPI_STATUS_IGNORE);
    record_->until_waits_
        .clear(); // drop any views we're keeping alive until wait()
  }

  // keep a reference to this view around until wait() is called
  template <typename View> void keep_until_wait(const View &v) {
    record_->until_waits_.push_back(std::make_shared<ViewHolder<View>>(v));
  }

private:
  std::shared_ptr<Record> record_;
};

} // namespace KokkosComm