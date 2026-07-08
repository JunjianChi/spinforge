#!/usr/bin/env bash
# The size-x-P strong-scaling grid and the energy comparison (the multiple-problem-sizes
# requirement and the joules-per-step idea). Produced the a800_size_scaling figure and the
# energy section of results/c2_a800.md.
#
#   Rental:  TORCHRUN=.venv/bin/torchrun ./scripts/run_size_energy.sh
#   Smoke:   SMOKE=1 TORCHRUN="uv run torchrun" ./scripts/run_size_energy.sh
set -uo pipefail  # NOT -e: per-step || guards record failures; the session must finish

if [ "${SMOKE:-0}" = "1" ]; then
  SIZES=${SIZES:-"8 12"}; PLIST=${PLIST:-2}; REPS=${REPS:-2}; WARMUP=${WARMUP:-1}
  DTYPE=${DTYPE:-float64}; ENERGY_N=${ENERGY_N:-8}; ENERGY_P=${ENERGY_P:-2}; ENERGY_REPS=${ENERGY_REPS:-2}
else
  SIZES=${SIZES:-"128 512"}; PLIST=${PLIST:-"2 4 8"}; REPS=${REPS:-20}; WARMUP=${WARMUP:-4}
  DTYPE=${DTYPE:-float32}; ENERGY_N=${ENERGY_N:-256}; ENERGY_P=${ENERGY_P:-8}; ENERGY_REPS=${ENERGY_REPS:-60}
fi
GATE="--gate-nx 8 --gate-ny 8 --gate-nz 8"
TORCHRUN=${TORCHRUN:-torchrun}
STAMP=sizes_$(date +%m%d_%H%M%S)
export PYTHONPATH=src:.
mkdir -p runs

b() { # P name extra...
  local P=$1 name=$2; shift 2
  mkdir -p "runs/c2_${STAMP}_P${P}_${name}"
  $TORCHRUN --standalone --nproc_per_node="$P" benchmarks/bench_dist_demag.py $GATE \
    --reps "$REPS" --warmup "$WARMUP" --dtype "$DTYPE" --modes fwd_adj \
    --run-name "c2_${STAMP}_P${P}_${name}" "$@" \
    > "runs/c2_${STAMP}_P${P}_${name}/console.log" 2>&1 || echo "!! $name P=$P"
  echo "== done P=$P $name"
}

echo "### size-x-P grid (r0 clean + optimized stack per size)"
for P in $PLIST; do
  for S in $SIZES; do
    b "$P" "strong${S}_r0" --nx "$S" --ny "$S" --nz "$S" --uninstrumented
    b "$P" "strong${S}_cum" --nx "$S" --ny "$S" --nz "$S" --use-rfft --schedule pipelined
  done
done

echo "### energy: mean draw sampled at 1 Hz around identical benchmark windows"
for row in r0 cum; do
  if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -l 1 \
      > "runs/power_${STAMP}_${row}.csv" 2>/dev/null &
    SMI=$!
  else
    SMI=""
  fi
  if [ "$row" = r0 ]; then EX="--uninstrumented"; else EX="--use-rfft --schedule pipelined"; fi
  mkdir -p "runs/c2_${STAMP}_P${ENERGY_P}_energy_${row}"
  $TORCHRUN --standalone --nproc_per_node="$ENERGY_P" benchmarks/bench_dist_demag.py $GATE \
    --nx "$ENERGY_N" --ny "$ENERGY_N" --nz "$ENERGY_N" --reps "$ENERGY_REPS" --warmup 4 \
    --dtype "$DTYPE" --modes fwd_adj $EX --run-name "c2_${STAMP}_P${ENERGY_P}_energy_${row}" \
    > "runs/c2_${STAMP}_P${ENERGY_P}_energy_${row}/console.log" 2>&1 || echo "!! energy $row"
  [ -n "$SMI" ] && kill $SMI 2>/dev/null
  echo "== done energy $row"
done
echo "### done: runs/c2_${STAMP}_* + runs/power_${STAMP}_*.csv"
