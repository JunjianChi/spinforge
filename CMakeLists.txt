# Shipped build for the native spinforge extensions (JIT cpp_extension.load is the dev loop).
# Requires a CUDA host; the MPI extension additionally needs a CUDA-aware MPI (see
# scripts/build_cuda_aware_ompi.sh). Configure with torch's cmake prefix and the venv python:
#
#   cmake -B build \
#     -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
#     -DPython3_EXECUTABLE="$(which python)" \
#     -DSPINFORGE_CUDA_ARCH=89        # sm_89 = RTX 4060; set to the node's GPU
#   cmake --build build && ctest --test-dir build
cmake_minimum_required(VERSION 3.18)
project(spinforge LANGUAGES CXX)

option(SPINFORGE_MPI "Build the CUDA-aware MPI collective extension" ON)
set(SPINFORGE_CUDA_ARCH "89" CACHE STRING "CMAKE_CUDA_ARCHITECTURES for the native kernels")

enable_testing()
add_subdirectory(src/spinforge/csrc)
