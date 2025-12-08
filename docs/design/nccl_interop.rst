*********************
NCCL interoperability
*********************

There are several challenges with supporting NCCL.

* For multi-process NCCL, we need a way to share a unique ID between processes so that the different processes know they're part of the same NCCL communicator. For the time being, this is accomplished via MPI in the nccl tests.

* NCCL has the concept of a non-blocking communicator. This causes all NCCL operations to potentially return ``ncclInProgress`` BEFORE they actually put an GPU operations into streams. This means we can't just synchronize on a CUDA stream in the NCCL backend's ``wait`` implementation. Either:

  * our NCCL operations need to effectively become blocking (checking NCCL's async status thing until it's no longer in progress)
  * our ``wait`` implementation needs to do that

* It is not guaranteed to be safe for NCCL operations and GPU-aware MPI operations to be simultaneously active on the same set of GPUs. This is a challenge both if we permit multiple backends to exist, and for interop with existing MPI and/or NCCL applications.
