# Native spinforge extensions: the single-GPU cuFFT demag + scale op, and (option SPINFORGE_MPI)
# the CUDA-aware MPI collectives. Torch's cmake package carries the ABI flags that must match the
# installed torch (mismatch = undefined-symbol crash at load -- the #1 pitfall).
enable_language(CUDA)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)  # cuFFT

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES ${SPINFORGE_CUDA_ARCH})

find_package(Python3 REQUIRED COMPONENTS Interpreter)
# torch/extension.h drags in Python.h; the JIT loader adds the interpreter's include dir
# automatically, a plain CMake build must do it itself (sysconfig is venv/uv-proof where
# find_package(Python3 ... Development) is not)
if(Python3_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['include'])"
    OUTPUT_VARIABLE SPINFORGE_PY_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

# single-GPU cuFFT demag + the scale op (validated via cpp_extension.load on the 4060)
add_library(spinforge_native SHARED demag.cu scale_op.cu)
target_include_directories(spinforge_native PRIVATE ${SPINFORGE_PY_INCLUDE})
target_link_libraries(spinforge_native PRIVATE "${TORCH_LIBRARIES}" CUDA::cufft)

if(SPINFORGE_MPI)
  # own MPI communicator inside the extension (MPI_Comm_dup) -- never torch's MPI backend
  find_package(MPI REQUIRED COMPONENTS CXX)
  add_library(spinforge_mpi SHARED mpi_collectives.cpp)
  target_include_directories(spinforge_mpi PRIVATE ${SPINFORGE_PY_INCLUDE})
  target_link_libraries(spinforge_mpi PRIVATE "${TORCH_LIBRARIES}" MPI::MPI_CXX)
endif()

# ctest smoke: the libraries must LOAD into torch (resolves the TORCH_LIBRARY registration and
# catches ABI mismatches). Pass -DPython3_EXECUTABLE=<venv python with torch installed>.
if(Python3_FOUND)
  add_test(
    NAME native_library_loads
    COMMAND
      ${Python3_EXECUTABLE} -c
      "import torch; torch.ops.load_library(r'$<TARGET_FILE:spinforge_native>'); print('ok')"
  )
  if(SPINFORGE_MPI)
    add_test(
      NAME mpi_library_loads
      COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${Python3_EXECUTABLE} -c
        "import mpi4py.MPI; import torch; torch.ops.load_library(r'$<TARGET_FILE:spinforge_mpi>'); print('ok')"
    )
  endif()
endif()
