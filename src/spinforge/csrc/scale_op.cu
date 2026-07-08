// Minimal CUDA custom op: proves the native build chain -- nvcc + torch link + ABI + dispatch.
// y = s * x, elementwise, double precision on CUDA. The backward is supplied in Python (native.py).
#include <torch/extension.h>

#include <cuda_runtime.h>

__global__ void scale_kernel(const double* x, double s, double* y, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = s * x[i];
}

torch::Tensor scale_forward(torch::Tensor x, double s) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat64, "x must be float64");
  auto xc = x.contiguous();
  auto y = torch::empty_like(xc);
  int64_t n = xc.numel();
  const int threads = 256;
  const int blocks = (int)((n + threads - 1) / threads);
  scale_kernel<<<blocks, threads>>>(
      xc.data_ptr<double>(), s, y.data_ptr<double>(), n);
  return y;
}

TORCH_LIBRARY(spinforge, m) {
  m.def("scale_forward(Tensor x, float s) -> Tensor", &scale_forward);
}
