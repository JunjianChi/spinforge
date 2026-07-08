#!/usr/bin/env bash
# Session 2: the true-speedup baseline, the capacity wall, and the cleanup measurements.
# Everything Session 1 left open, sequenced; ~1-1.5 h on the node. Pure execution.
#
#   Rental:  ./scripts/run_session2.sh
#   Smoke:   SMOKE=1 TORCHRUN="uv run torchrun" PY="uv run python" ./scripts/run_session2.sh
set -uo pipefail  # NOT -e: later sections must run even if an unguarded step crashes (a recorded OOM already exits 0)

if [ "${SMOKE:-0}" = "1" ]; then
  SINGLES=${SINGLES:-"8 12"}; BASE=${BASE:-8}; BASE_NZ=${BASE_NZ:-8}
  REPS=${REPS:-2}; WARMUP=${WARMUP:-1}; DTYPE=${DTYPE:-float64}; PMAX=${PMAX:-2}
  SWEEP=${SWEEP:-}; PROBE=${PROBE:-}
else
  SINGLES=${SINGLES:-"128 192 256"}; BASE=${BASE:-256}; BASE_NZ=${BASE_NZ:-256}
  REPS=${REPS:-20}; WARMUP=${WARMUP:-4}; DTYPE=${DTYPE:-float32}; PMAX=${PMAX:-8}
  SWEEP=${SWEEP:-"192,256,320"}; PROBE=${PROBE:-}  # PROBE auto-derived from the fit when empty
fi
GATE="--gate-nx 8 --gate-ny 8 --gate-nz 8"
TORCHRUN=${TORCHRUN:-torchrun}
PY=${PY:-python}
command -v ${TORCHRUN%% *} >/dev/null || { echo "torchrun not found: set TORCHRUN" >&2; exit 1; }
STAMP=$(date +%m%d_%H%M%S)
export PYTHONPATH=src:.
mkdir -p runs
{ nvidia-smi && nvidia-smi topo -m; } > "runs/env_s2_${STAMP}.txt" 2>&1 || true
echo "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-unset}" >> "runs/env_s2_${STAMP}.txt"

bench() { # P name extra...
  local P=$1 name=$2; shift 2
  echo "== P=$P  $name  $*"
  mkdir -p "runs/c2_${STAMP}_P${P}_${name}"
  $TORCHRUN --standalone --nproc_per_node="$P" benchmarks/bench_dist_demag.py $GATE \
    --reps "$REPS" --warmup "$WARMUP" --dtype "$DTYPE" \
    --run-name "c2_${STAMP}_P${P}_${name}" "$@" \
    > "runs/c2_${STAMP}_P${P}_${name}/console.log" 2>&1 || echo "  !! $name failed (log kept)"
}

echo "### 1) uninstrumented reruns (clean per-row numbers for the report table)"
bench "$PMAX" strong_rfft_clean --nx "$BASE" --ny "$BASE" --nz "$BASE_NZ" --use-rfft --uninstrumented
bench "$PMAX" strong_mixed_clean --nx "$BASE" --ny "$BASE" --nz "$BASE_NZ" --mixed-wire --dtype float64
bench "$PMAX" strong_r0_clean --nx "$BASE" --ny "$BASE" --nz "$BASE_NZ" --uninstrumented

echo "### 2) single-device baselines (the true-speedup denominator)"
for S in $SINGLES; do
  mkdir -p "runs/c2single_${STAMP}_${S}"
  $PY benchmarks/bench_single.py --nx "$S" --ny "$S" --nz "$S" --dtype "$DTYPE" \
    --reps "$REPS" --warmup "$WARMUP" --run-name "c2single_${STAMP}_${S}" \
    > "runs/c2single_${STAMP}_${S}/console.log" 2>&1 || echo "  !! single $S failed"
  echo "== single ${S}^3 done"
done

echo "### 3) distributed-on-one-rank (the cost of going distributed, NOT a speedup denominator)"
bench 1 strong_r0_p1 --nx "$BASE" --ny "$BASE" --nz "$BASE_NZ" --uninstrumented

if [ -n "$SWEEP" ]; then
  echo "### 4) capacity wall: sweep -> fit -> predict -> probe both sides"
  $PY benchmarks/mem_wall.py --sweep "$SWEEP" --dtype "$DTYPE" \
    > "runs/memwall_${STAMP}.json" 2>&1 || echo "  !! sweep failed"
  WALL=$(grep -o '"predicted_wall_side": [0-9]*' "runs/memwall_${STAMP}.json" | grep -o '[0-9]*' || echo "")
  echo "== predicted wall side: ${WALL:-unknown}"
  if [ -n "${WALL}" ]; then
    OVER=${PROBE:-$(( (WALL / 32 + 2) * 32 ))}  # just past the wall, divisibility-friendly
    echo "== probing single GPU at ${OVER}^3 (expect OOM -- the wall's boundary evidence)"
    $PY benchmarks/mem_wall.py --probe "$OVER" --dtype "$DTYPE" \
      >> "runs/memwall_${STAMP}.json" 2>&1
    echo "== same grid, P=$PMAX distributed (expect success + timing)"
    bench "$PMAX" overwall --nx "$OVER" --ny "$OVER" --nz "$OVER" --modes fwd_adj --uninstrumented
  fi
fi

echo "### 5) nsys traces (best effort; wall+CI carry the claim if unavailable)"
if ! command -v nsys > /dev/null; then
  apt-get install -y -q nsight-systems-cli > /dev/null 2>&1 || echo "  !! nsys unavailable (recorded)"
fi
if command -v nsys > /dev/null; then
  for sched in sequential pipelined; do
    nsys profile -o "runs/nsys_${STAMP}_${sched}" --force-overwrite true \
      $TORCHRUN --standalone --nproc_per_node="$PMAX" benchmarks/bench_dist_demag.py $GATE \
      --nx "$BASE" --ny "$BASE" --nz "$BASE_NZ" --reps 3 --warmup 1 --modes fwd_adj \
      --uninstrumented --schedule "$sched" > /dev/null 2>&1 || echo "  !! nsys $sched failed"
    echo "== nsys $sched done"
  done
fi

echo "### done: runs/c2_${STAMP}_*, runs/c2single_${STAMP}_*, runs/memwall_${STAMP}.json -- scp home"
