// Mixed precision: the Newell demag as a CUDA cuFFT convolution, float64 (Z2Z) or float32 (C2C).
// The kernel (FFT of the real even Newell tensor, 6 components) is built once in PyTorch and passed
// in; this op is the convolution engine: pad -> cuFFT forward -> contract -> cuFFT inverse ->
// normalize + crop. Matches core.demag.DemagField; self-adjoint, so the Python backward reuses it.
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include <c10/cuda/CUDAStream.h>

#include <memory>

#define CUFFT_CHECK(x) TORCH_CHECK((x) == CUFFT_SUCCESS, "cuFFT error ", (int)(x))
#define CUDA_CHECK(x)                                                                 \
  do {                                                                                \
    cudaError_t err_ = (x);                                                           \
    TORCH_CHECK(err_ == cudaSuccess, "CUDA error: ", cudaGetErrorString(err_));       \
  } while (0)
// check the last kernel launch (invalid config etc.); async execution errors surface at the sync
#define CUDA_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

// RAII owners so the raw cudaMalloc buffers + the cuFFT plan are released on ANY unwind path
// (an OOM or a cuFFT failure between allocation and the manual free would otherwise leak the
// largest buffers in the program). No raw owning pointers (CLAUDE.md C++ rule).
struct CudaDeleter {
  void operator()(void* p) const noexcept { cudaFree(p); }
};
template <typename T>
using cuda_ptr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
static cuda_ptr<T> cuda_alloc(long n) {
  void* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  return cuda_ptr<T>(static_cast<T*>(p));
}

struct CufftPlan {
  cufftHandle handle = 0;
  bool valid = false;
  ~CufftPlan() {
    if (valid) cufftDestroy(handle);
  }
};

// precision-overloaded cuFFT plan/exec so the impl templates cleanly on the complex type
static cufftResult fft_plan(cufftHandle* p, int x, int y, int z, cufftDoubleComplex*) {
  return cufftPlan3d(p, x, y, z, CUFFT_Z2Z);
}
static cufftResult fft_plan(cufftHandle* p, int x, int y, int z, cufftComplex*) {
  return cufftPlan3d(p, x, y, z, CUFFT_C2C);
}
static cufftResult fft_exec(cufftHandle p, cufftDoubleComplex* a, cufftDoubleComplex* b, int d) {
  return cufftExecZ2Z(p, a, b, d);
}
static cufftResult fft_exec(cufftHandle p, cufftComplex* a, cufftComplex* b, int d) {
  return cufftExecC2C(p, a, b, d);
}

template <typename C>
__global__ void zero_c(C* a, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { a[i].x = 0; a[i].y = 0; }
}

template <typename C, typename R>
__global__ void pad_copy(const R* src, C* dst, R ms, int nx, int ny, int nz, int Ny, int Nz) {
  long n = (long)nx * ny * nz, idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int iz = idx % nz, iy = (idx / nz) % ny, ix = idx / ((long)nz * ny);
  long o = ((long)ix * Ny + iy) * Nz + iz;
  dst[o].x = ms * src[idx];
  dst[o].y = 0;
}

template <typename C, typename R>
__global__ void contract(const C* mx, const C* my, const C* mz, const R* Kxx, const R* Kxy,
                         const R* Kxz, const R* Kyy, const R* Kyz, const R* Kzz, C* hx, C* hy,
                         C* hz, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  R xr = mx[i].x, xi = mx[i].y, yr = my[i].x, yi = my[i].y, zr = mz[i].x, zi = mz[i].y;
  hx[i].x = Kxx[i] * xr + Kxy[i] * yr + Kxz[i] * zr;
  hx[i].y = Kxx[i] * xi + Kxy[i] * yi + Kxz[i] * zi;
  hy[i].x = Kxy[i] * xr + Kyy[i] * yr + Kyz[i] * zr;
  hy[i].y = Kxy[i] * xi + Kyy[i] * yi + Kyz[i] * zi;
  hz[i].x = Kxz[i] * xr + Kyz[i] * yr + Kzz[i] * zr;
  hz[i].y = Kxz[i] * xi + Kyz[i] * yi + Kzz[i] * zi;
}

template <typename C, typename R>
__global__ void crop_real(const C* src, R* dst, R inv, int nx, int ny, int nz, int Ny, int Nz) {
  long n = (long)nx * ny * nz, idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int iz = idx % nz, iy = (idx / nz) % ny, ix = idx / ((long)nz * ny);
  long o = ((long)ix * Ny + iy) * Nz + iz;
  dst[idx] = src[o].x * inv;
}

template <typename C, typename R>
static torch::Tensor demag_impl(torch::Tensor m, torch::Tensor kernel, double ms, int64_t nx,
                                int64_t ny, int64_t nz) {
  int Nx = 2 * nx, Ny = 2 * ny, Nz = 2 * nz;
  long Ntot = (long)Nx * Ny * Nz, ntot = (long)nx * ny * nz;
  m = m.contiguous();
  auto K = kernel.contiguous();
  const R* Kp[6];
  for (int k = 0; k < 6; k++) Kp[k] = K[k].data_ptr<R>();
  cuda_ptr<C> mf[3], hf[3];
  for (int a = 0; a < 3; a++) {
    mf[a] = cuda_alloc<C>(Ntot);
    hf[a] = cuda_alloc<C>(Ntot);
  }
  int thr = 256;
  long gN = (Ntot + thr - 1) / thr, gn = (ntot + thr - 1) / thr;
  auto stream = c10::cuda::getCurrentCUDAStream(m.device().index());
  for (int a = 0; a < 3; a++) {
    zero_c<C><<<gN, thr, 0, stream>>>(mf[a].get(), Ntot);
    CUDA_LAUNCH_CHECK();
    auto ma = m.select(3, a).contiguous();
    pad_copy<C, R><<<gn, thr, 0, stream>>>(ma.data_ptr<R>(), mf[a].get(), (R)ms, nx, ny, nz, Ny, Nz);
    CUDA_LAUNCH_CHECK();
  }
  CufftPlan plan;
  CUFFT_CHECK(fft_plan(&plan.handle, Nx, Ny, Nz, (C*)nullptr));
  CUFFT_CHECK(cufftSetStream(plan.handle, stream));
  plan.valid = true;
  for (int a = 0; a < 3; a++)
    CUFFT_CHECK(fft_exec(plan.handle, mf[a].get(), mf[a].get(), CUFFT_FORWARD));
  contract<C, R><<<gN, thr, 0, stream>>>(mf[0].get(), mf[1].get(), mf[2].get(), Kp[0], Kp[1],
                                         Kp[2], Kp[3], Kp[4], Kp[5], hf[0].get(), hf[1].get(),
                                         hf[2].get(), Ntot);
  CUDA_LAUNCH_CHECK();
  for (int a = 0; a < 3; a++)
    CUFFT_CHECK(fft_exec(plan.handle, hf[a].get(), hf[a].get(), CUFFT_INVERSE));
  auto h = torch::empty({nx, ny, nz, 3}, m.options());
  R inv = (R)(1.0 / (double)Ntot);
  for (int a = 0; a < 3; a++) {
    auto ha = torch::empty({nx, ny, nz}, m.options());
    crop_real<C, R><<<gn, thr, 0, stream>>>(hf[a].get(), ha.data_ptr<R>(), inv, nx, ny, nz, Ny, Nz);
    CUDA_LAUNCH_CHECK();
    h.select(3, a).copy_(ha);
  }
  // drain THIS stream before the RAII buffers/plan free at scope exit (all async work done)
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return h;
}

torch::Tensor demag_forward(torch::Tensor m, torch::Tensor kernel, double ms, int64_t nx,
                            int64_t ny, int64_t nz) {
  TORCH_CHECK(m.is_cuda(), "m must be a CUDA tensor");
  // every axis is padded to 2*n; a singleton axis would make the passed kernel (built by DemagField,
  // which collapses singleton axes) disagree with the FFT extent and read out of bounds.
  TORCH_CHECK(nx > 1 && ny > 1 && nz > 1, "native demag requires all mesh dims > 1 (got ", nx, ",",
              ny, ",", nz, "); use the PyTorch DemagField for quasi-2D meshes");
  TORCH_CHECK(kernel.scalar_type() == m.scalar_type(), "kernel and m must share dtype");
  if (m.scalar_type() == torch::kFloat64)
    return demag_impl<cufftDoubleComplex, double>(m, kernel, ms, nx, ny, nz);
  TORCH_CHECK(m.scalar_type() == torch::kFloat32, "demag supports float32/float64 only");
  return demag_impl<cufftComplex, float>(m, kernel, ms, nx, ny, nz);
}

TORCH_LIBRARY_FRAGMENT(spinforge, m) {
  m.def("demag_forward(Tensor m, Tensor kernel, float ms, int nx, int ny, int nz) -> Tensor",
        &demag_forward);
}
