#!/usr/bin/env bash
# Offline-phase v2 comparison sweep (docs/offline_training_design.md SS4a):
# baseline vs. forced-early/forced-late one-shot triggers vs. periodic
# consolidation, run one job at a time.  The GPU simulator is not bitwise
# deterministic, so the second sample per condition reruns the identical
# command with the same seed.
#
# Usage: PY=/path/to/python ./run_offline_v2_sweep.sh   (default: python on PATH)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PY:-python}"
export MUJOCO_GL=egl
STEPS=60000
OUT_ROOT=outputs/vec
LOGDIR=outputs/vec/offline_v2_logs
mkdir -p "$LOGDIR"

BASE_FLAGS=(--distill --sysid --terminal_value --steps "$STEPS")

# Fixed order (bash assoc arrays don't preserve insertion order).
ORDER=(baseline forced_early forced_late periodic)
declare -A COND_FLAGS=(
  [baseline]=""
  [forced_early]="--offline_force_step 20000 --offline_mode once"
  [forced_late]="--offline_force_step 45000 --offline_mode once"
  [periodic]="--offline_start_val_loss 0.15 --offline_confirm_checks 2 --offline_mode periodic --offline_periodic_num_updates 8000"
)

sweep_log="$LOGDIR/sweep.log"
echo "=== sweep start $(date '+%F %T') ===" | tee -a "$sweep_log"

declare -A RUNS=(
  # Baseline sample 1 is outputs/vec/dagger_b0, so only r2 runs here.
  [baseline]="2"
  [forced_early]="1 2"
  [forced_late]="1 2"
  [periodic]="1 2"
)

for tag in "${ORDER[@]}"; do
  # shellcheck disable=SC2206
  extra_flags=(${COND_FLAGS[$tag]})
  for run in ${RUNS[$tag]}; do
    name="offline_v2_${tag}_r${run}"
    log="$LOGDIR/${name}.log"
    echo "--- [$(date '+%F %T')] starting $name ---" | tee -a "$sweep_log"
    "$PY" scripts/train_vec.py --name "$name" --out_root "$OUT_ROOT" \
      "${BASE_FLAGS[@]}" "${extra_flags[@]}" > "$log" 2>&1
    status=$?
    echo "--- [$(date '+%F %T')] finished $name (exit $status) ---" | tee -a "$sweep_log"
  done
done

echo "=== sweep done $(date '+%F %T') ===" | tee -a "$sweep_log"
