#!/usr/bin/env bash
# Build a CUDA-aware OpenMPI (UCX --with-cuda, then OpenMPI --with-cuda --with-ucx) into $PREFIX.
# Needed for spinforge's native GPU-direct collectives: the distro OpenMPI is NOT CUDA-aware.
# Works on the single-GPU dev box and on a rented multi-GPU node alike (~30-60 min).
#
# Usage:   CUDA_HOME=/usr/local/cuda-13.0 ./scripts/build_cuda_aware_ompi.sh
# Then:    export PATH="$PREFIX/bin:$PATH"; export LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"
#          MPICC="$PREFIX/bin/mpicc" python -m pip install --no-cache-dir mpi4py
# Verify:  ompi_info --parsable --all | grep -i mpi_built_with_cuda_support:value
set -euo pipefail

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
PREFIX=${PREFIX:-$HOME/opt/ompi-cuda}
# UCX must be recent enough for the CUDA toolkit's driver-API headers (1.18 fails against
# CUDA 13: PFN_cuMemGetHandleForAddressRange); 1.21.0 is the newest stable as of 2026-07.
UCX_VER=${UCX_VER:-1.21.0}
OMPI_VER=${OMPI_VER:-5.0.10}
JOBS=${JOBS:-$(nproc)}
WORK=${WORK:-$HOME/build-ompi-cuda}

[ -d "$CUDA_HOME" ] || { echo "CUDA_HOME=$CUDA_HOME not found" >&2; exit 1; }
mkdir -p "$WORK" "$PREFIX"
cd "$WORK"

echo "== UCX $UCX_VER (CUDA transport) =="
if [ ! -x "$PREFIX/bin/ucx_info" ]; then
  curl -fsSLO "https://github.com/openucx/ucx/releases/download/v${UCX_VER}/ucx-${UCX_VER}.tar.gz"
  tar xf "ucx-${UCX_VER}.tar.gz"
  cd "ucx-${UCX_VER}"
  ./configure --prefix="$PREFIX" --with-cuda="$CUDA_HOME" --enable-mt
  make -j "$JOBS"
  make install
  cd "$WORK"
fi
"$PREFIX/bin/ucx_info" -v | head -2  # hard-fails the script if the UCX build did not land

echo "== OpenMPI $OMPI_VER (CUDA-aware, over UCX) =="
if [ ! -x "$PREFIX/bin/mpicc" ]; then
  OMPI_SERIES=$(echo "$OMPI_VER" | cut -d. -f1-2)
  curl -fsSLO "https://download.open-mpi.org/release/open-mpi/v${OMPI_SERIES}/openmpi-${OMPI_VER}.tar.bz2"
  tar xf "openmpi-${OMPI_VER}.tar.bz2"
  cd "openmpi-${OMPI_VER}"
  ./configure --prefix="$PREFIX" --with-cuda="$CUDA_HOME" --with-ucx="$PREFIX"
  make -j "$JOBS"
  make install
  cd "$WORK"
fi

echo "== verify =="
"$PREFIX/bin/ompi_info" --parsable --all | grep -i "mpi_built_with_cuda_support:value" || true
echo "done: PREFIX=$PREFIX"
