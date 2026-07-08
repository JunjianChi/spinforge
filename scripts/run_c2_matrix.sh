#!/usr/bin/env bash
# Measurement-matrix driver: the rental session's main loop, mechanical by design.
# Rows x columns land as provenance-stamped summary JSONs under runs/ (scp them home as they
# appear). Machine parameters (alpha/beta) are measured first; the pre-registered naive-schedule
# baseline equivalence pair runs before any row is quoted; perturbation bounds close the set.
#
#   Rental:  LINKBW=<measured GB/s> ALPHA_US=<from the ab_ runs> ./scripts/run_c2_matrix.sh
#   Smoke:   SMOKE=1 ./scripts/run_c2_matrix.sh          (CPU/gloo, tiny -- wiring check only)
#
# Rows: r0 (naive) / rfft / mixed-wire / overlap (pipelined) / rfft+overlap (cumulative; the
# mixed+async composition is a pinned follow-up) / no-comm + no-fft (high-watermark bounds).
# Columns: strong (fixed global grid, P swept) and weak (fixed per-rank slab); fwd vs fwd+adjoint
# vs +checkpoint are the --modes within every run.
set -euo pipefail

if [ "${SMOKE:-0}" = "1" ]; then
  PLIST=${PLIST:-"2"}; BASE=${BASE:-8}; BASE_NZ=${BASE_NZ:-8}; WEAK_NZ=${WEAK_NZ:-4}
  REPS=${REPS:-2}; WARMUP=${WARMUP:-1}; DTYPE=${DTYPE:-float64}
  AB_SIZES=${AB_SIZES:-"64,256"}  # KB; the smoke checks wiring, not bandwidth
else
  PLIST=${PLIST:-"2 4 8"}; BASE=${BASE:-256}; BASE_NZ=${BASE_NZ:-256}; WEAK_NZ=${WEAK_NZ:-32}
  REPS=${REPS:-30}; WARMUP=${WARMUP:-5}; DTYPE=${DTYPE:-float32}
fi
LINKBW=${LINKBW:-}    # GB/s, MEASURED per-rank ejection bandwidth (empty -> no roofline column)
AB_SIZES=${AB_SIZES:-"16384,65536,262144"}  # KB; large enough to clear the launch-jitter floor on fast fabrics
ALPHA_US=${ALPHA_US:-}  # us/message from the ab_* fits (empty -> bytes-only bound)
GATE="--gate-nx 8 --gate-ny 8 --gate-nz 8"
BENCH="benchmarks/bench_dist_demag.py"
TORCHRUN=${TORCHRUN:-torchrun}  # e.g. TORCHRUN="uv run torchrun" on a uv-managed machine
command -v ${TORCHRUN%% *} >/dev/null || { echo "torchrun not found: set TORCHRUN" >&2; exit 1; }
STAMP=$(date +%m%d_%H%M%S)
PMAX=$(echo "$PLIST" | awk '{print $NF}')
export PYTHONPATH=src:.

run() { # P name extra-args...
  local P=$1 name=$2; shift 2
  echo "== P=$P  $name  $*"
  pre "c2_${STAMP}_P${P}_${name}"
  $TORCHRUN --standalone --nproc_per_node="$P" $BENCH $GATE \
    --reps "$REPS" --warmup "$WARMUP" --dtype "$DTYPE" \
    ${LINKBW:+--link-bw-gbs "$LINKBW"} ${ALPHA_US:+--alpha-us "$ALPHA_US"} \
    --run-name "c2_${STAMP}_P${P}_${name}" "$@" > "runs/c2_${STAMP}_P${P}_${name}/console.log" 2>&1
}

pre() { mkdir -p "runs/$1"; }  # the log target must exist before torchrun opens it

mkdir -p runs
{ nvidia-smi && nvidia-smi topo -m; } > "runs/env_${STAMP}.txt" 2>&1 || echo "no GPU (smoke)" > "runs/env_${STAMP}.txt"

echo "### 0) alpha-beta machine parameters (per P)"
for P in $PLIST; do
  [ "$P" -lt 2 ] && continue
  pre "ab_${STAMP}_P${P}"
  $TORCHRUN --standalone --nproc_per_node="$P" benchmarks/bench_alpha_beta.py \
    --sizes-kb "$AB_SIZES" --reps "$REPS" --warmup "$WARMUP" --run-name "ab_${STAMP}_P${P}" \
    > "runs/ab_${STAMP}_P${P}/console.log" 2>&1 \
    || echo "  !! alpha-beta fit failed at P=$P (console.log kept); matrix continues"
done

echo "### 1) pre-registered naive-baseline equivalence (instrumented blocking vs default async, P=$PMAX)"
run "$PMAX" r0equiv_instr --nx "$BASE" --ny "$BASE" --nz "$BASE_NZ" --modes fwd_adj
run "$PMAX" r0equiv_async --nx "$BASE" --ny "$BASE" --nz "$BASE_NZ" --modes fwd_adj --uninstrumented

echo "### 2) the matrix: rows x strong/weak"
ROWS=(
  "r0:"
  "rfft:--use-rfft"
  "mixed:--mixed-wire --dtype float64"  # f32 wire vs f64 compute: a real halving (at f32 compute the cast is identity)
  "overlap:--schedule pipelined"
  "cum_rfft_overlap:--use-rfft --schedule pipelined"
)
for P in $PLIST; do
  for row in "${ROWS[@]}"; do
    name="${row%%:*}"; extra="${row#*:}"
    # strong: fixed global grid
    run "$P" "strong_${name}" --nx "$BASE" --ny "$BASE" --nz "$BASE_NZ" $extra
    # weak: fixed per-rank slab (nz scales with P)
    run "$P" "weak_${name}" --nx "$BASE" --ny "$BASE" --nz "$((WEAK_NZ * P))" $extra
  done
done

echo "### 3) high-watermark bounds (comm-dominance by elimination, P=$PMAX, strong grid)"
run "$PMAX" bound_nocomm --nx "$BASE" --ny "$BASE" --nz "$BASE_NZ" --modes fwd_adj --perturb no-comm
run "$PMAX" bound_nofft --nx "$BASE" --ny "$BASE" --nz "$BASE_NZ" --modes fwd_adj --perturb no-fft

echo "### done: summaries under runs/c2_${STAMP}_* and runs/ab_${STAMP}_* -- scp them home"
