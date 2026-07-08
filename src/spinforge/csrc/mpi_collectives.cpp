// From-scratch HPC piece: CUDA-aware MPI all-to-all collectives, GPU-direct (no host
// staging). MPI is initialized by mpi4py / mpirun; the extension duplicates MPI_COMM_WORLD once
// into its OWN communicator so library collectives can never interleave with (and deadlock
// against) the application's own MPI traffic.
//
// Three variants, one contract (all-to-all over dim 0, equal chunks):
//   mpi_all_to_all          -- blocking MPI_Alltoall: the shipped op (sequential schedule).
//   mpi_ialltoall           -- MPI_Ialltoall launched + waited immediately: the CUDA-awareness
//                              PROBE for the nonblocking path (stock OpenMPI's libnbc is NOT
//                              CUDA-aware; this op answers that question per machine before any
//                              overlap variant is built on it).
//   mpi_pairwise_all_to_all -- MPI_Isend/Irecv per peer + Waitall: the fallback when Ialltoall
//                              fails the probe (point-to-point goes through UCX, which IS
//                              CUDA-aware); also the building block a hand-rolled overlapped
//                              schedule would use.
//
// Compiled + correctness-tested with 2 ranks sharing one GPU (MPI has no NCCL-style
// duplicate-GPU restriction; same-device transfers go through CUDA IPC). Performance numbers
// are only meaningful on a real multi-GPU node.
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <mpi.h>

#include <vector>

namespace {

// Lazily duplicate MPI_COMM_WORLD once. MPI_Comm_dup is collective: every rank calls it on the
// first spinforge op, which all ranks execute symmetrically.
MPI_Comm spinforge_comm() {
  static MPI_Comm comm = [] {
    int initialized = 0;
    MPI_Initialized(&initialized);
    TORCH_CHECK(initialized, "MPI not initialized (import mpi4py.MPI / launch with mpirun)");
    MPI_Comm c = MPI_COMM_NULL;
    TORCH_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &c) == MPI_SUCCESS, "MPI_Comm_dup failed");
    return c;
  }();
  return comm;
}

struct A2AArgs {
  MPI_Comm comm;
  int world;
  int rank;
  int64_t per;      // elements sent to each peer
  MPI_Datatype dt;  // element datatype
};

// Shared validation + the stream drain every variant needs: CUDA-aware MPI issues its transfers
// on its own streams, unordered against torch's current stream, so without the drain it can read
// x (a cuFFT/contract output) before the producing kernels finish -> a stale/partial transpose ->
// wrong forward, and (the adjoint reuses these ops) a wrong gradient.
A2AArgs prepare(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
  MPI_Comm comm = spinforge_comm();
  A2AArgs a{comm, 0, 0, 0, MPI_DATATYPE_NULL};
  MPI_Comm_size(comm, &a.world);
  MPI_Comm_rank(comm, &a.rank);
  TORCH_CHECK(x.size(0) % a.world == 0, "dim 0 (", x.size(0),
              ") must be divisible by world size (", a.world, ")");
  // guard the dtype: otherwise anything not float64 is silently sent as MPI_FLOAT, so a complex
  // or half tensor moves the wrong byte count (view complex as real before calling, as the
  // Python wrapper does)
  TORCH_CHECK(x.scalar_type() == torch::kFloat64 || x.scalar_type() == torch::kFloat32,
              "mpi all_to_all supports float32/float64 only");
  a.dt = (x.scalar_type() == torch::kFloat64) ? MPI_DOUBLE : MPI_FLOAT;
  a.per = x.numel() / a.world;
  TORCH_CHECK(a.per <= INT_MAX, "per-rank chunk (", a.per, " elements) exceeds the int MPI count (use MPI_Type or chunking)");
  c10::cuda::getCurrentCUDAStream(x.device().index()).synchronize();
  return a;
}

}  // namespace

torch::Tensor mpi_all_to_all(torch::Tensor x) {
  A2AArgs a = prepare(x);
  auto out = torch::empty_like(x);
  int err = MPI_Alltoall(x.data_ptr(), (int)a.per, a.dt, out.data_ptr(), (int)a.per, a.dt, a.comm);
  TORCH_CHECK(err == MPI_SUCCESS, "MPI_Alltoall failed: ", err);
  return out;
}

torch::Tensor mpi_ialltoall(torch::Tensor x) {
  A2AArgs a = prepare(x);
  auto out = torch::empty_like(x);
  MPI_Request req = MPI_REQUEST_NULL;
  int err = MPI_Ialltoall(x.data_ptr(), (int)a.per, a.dt, out.data_ptr(), (int)a.per, a.dt,
                          a.comm, &req);
  TORCH_CHECK(err == MPI_SUCCESS, "MPI_Ialltoall failed: ", err);
  err = MPI_Wait(&req, MPI_STATUS_IGNORE);
  TORCH_CHECK(err == MPI_SUCCESS, "MPI_Wait failed: ", err);
  return out;
}

torch::Tensor mpi_pairwise_all_to_all(torch::Tensor x) {
  A2AArgs a = prepare(x);  // validate + drain FIRST; x is fully produced past this point
  auto out = torch::empty_like(x);
  // The self-chunk never touches MPI: enqueue a stream-ordered device copy. It aliases nothing
  // MPI reads or writes (peer chunks only), so it safely overlaps the transfers below and is
  // ordered before any downstream consumer of `out` on torch's stream.
  const int64_t rows_per = x.size(0) / a.world;
  out.narrow(0, a.rank * rows_per, rows_per).copy_(x.narrow(0, a.rank * rows_per, rows_per));

  auto* src = static_cast<char*>(x.data_ptr());
  auto* dst = static_cast<char*>(out.data_ptr());
  const size_t bytes = static_cast<size_t>(a.per) * x.element_size();
  std::vector<MPI_Request> reqs;
  reqs.reserve(2 * (a.world - 1));
  for (int peer = 0; peer < a.world; ++peer) {  // receives first: no ring deadlock by structure
    if (peer == a.rank) continue;
    reqs.emplace_back();
    int err = MPI_Irecv(dst + peer * bytes, (int)a.per, a.dt, peer, 0, a.comm, &reqs.back());
    TORCH_CHECK(err == MPI_SUCCESS, "MPI_Irecv failed: ", err);
  }
  for (int peer = 0; peer < a.world; ++peer) {
    if (peer == a.rank) continue;
    reqs.emplace_back();
    int err = MPI_Isend(src + peer * bytes, (int)a.per, a.dt, peer, 0, a.comm, &reqs.back());
    TORCH_CHECK(err == MPI_SUCCESS, "MPI_Isend failed: ", err);
  }
  int err = MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
  TORCH_CHECK(err == MPI_SUCCESS, "MPI_Waitall failed: ", err);
  return out;
}

TORCH_LIBRARY(spinforge_mpi, m) {
  m.def("mpi_all_to_all(Tensor x) -> Tensor", &mpi_all_to_all);
  m.def("mpi_ialltoall(Tensor x) -> Tensor", &mpi_ialltoall);
  m.def("mpi_pairwise_all_to_all(Tensor x) -> Tensor", &mpi_pairwise_all_to_all);
}
