#!/usr/bin/env bash
# Reruns every comfree_tdmpc training experiment in EXPERIMENTS.md twice more
# (rep2, rep3; the original run counts as rep1) for variance estimation.  The
# flags are copied from each original run's config.json.
#
# Not repeated: the throughput/verification benchmarks (06/07/08), the
# eval-only rollout-budget ablation (rerun it against the new checkpoints if
# needed), and outputs/pilot/pilot, whose launch command is not recorded.
# The offline_v2 sweep is topped up to three identically configured samples
# per condition (offline_v2_<cond>_r{1,2,3}); dagger_b0 is not reused as a
# baseline sample because it additionally sets --actor_mode bc_mu.
#
# Jobs run strictly one at a time: 16 vec-envs plus the planner already fill
# one GPU, so concurrent runs would only split its throughput.  At 150k steps
# expect roughly 24-30 h per rep; run it under tmux/screen/nohup.
#
# Resumable: a job is skipped when its config.json (written only at the end of
# a run) already exists; run_ablation.py applies the same rule per rung.
#
# Usage: PY=/path/to/python ./rerun_ablation_variance.sh   (default: python on PATH)

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PY:-python}"
export MUJOCO_GL=egl

REPS=(2 3)              # the original run on disk is rep1
OFFLINE_V2_TARGET_REPS=3

DRIVER_LOG="outputs/variance_reruns/driver.log"
mkdir -p "$(dirname "$DRIVER_LOG")"

SECONDS=0
N_OK=0
N_SKIP=0
N_FAIL=0
FAILED_JOBS=()

fmt_dur() { printf '%dh%02dm%02ds' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }
log() { echo "[$(date '+%F %T')] [+$(fmt_dur "$SECONDS")] $*" | tee -a "$DRIVER_LOG"; }
gpu_status() { nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"; }

# run_train <name> <out_root> <train_vec.py args...>
run_train() {
  local name="$1" out_root="$2"; shift 2
  # config.json is written only at the end of a run; agent.pt is also saved periodically mid-run.
  local ckpt="$out_root/$name/config.json"
  if [[ -f "$ckpt" ]]; then
    log "SKIP   $name (config.json already present, run finished)"
    N_SKIP=$((N_SKIP+1)); return 0
  fi
  mkdir -p "$out_root"
  local log_file="$out_root/${name}.log"
  log "START  $name  [gpu: $(gpu_status)]"
  local t0=$SECONDS
  "$PY" scripts/train_vec.py --name "$name" --out_root "$out_root" --tag "$name" "$@" \
    > "$log_file" 2>&1
  local rc=$? dt=$((SECONDS - t0))
  if [[ $rc -eq 0 ]]; then
    log "OK     $name  ($(fmt_dur "$dt"))"
    N_OK=$((N_OK+1))
  else
    log "FAILED $name  (exit $rc, $(fmt_dur "$dt")) -- see $log_file"
    N_FAIL=$((N_FAIL+1)); FAILED_JOBS+=("$name")
  fi
}

# run_ablation <label> <out_root> <steps> <only rungs...>
# run_ablation.py skips rungs whose config.json already exists.
run_ablation() {
  local label="$1" out_root="$2" steps="$3"; shift 3
  log "START  ablation:$label -> $out_root  [gpu: $(gpu_status)]"
  local t0=$SECONDS
  "$PY" scripts/run_ablation.py --steps "$steps" --out_root "$out_root" --only "$@" \
    2>&1 | tee -a "$DRIVER_LOG"
  local dt=$((SECONDS - t0))
  log "DONE   ablation:$label  ($(fmt_dur "$dt"))"
}

log "=== variance rerun sweep start === reps: ${REPS[*]}"
log "GPU at start: $(gpu_status)"

for rep in "${REPS[@]}"; do
  log "--- rep $rep ---"

  # 1. A-F ablation ladder (F is launched as a separate --only call)
  run_ablation "A-E" "outputs/ablation_rep${rep}"     150000 a_base b_distill c_sysid d_termq e_bc
  run_ablation "F"   "outputs/ablation_rep${rep}"     150000 f_meantarget

  # 2. d_termq / e_bc rerun at 300k steps
  run_ablation "300k" "outputs/ablation_300k_rep${rep}" 300000 d_termq e_bc

  # 3. Net-size sweep
  run_train "netsize_small_rep${rep}"   outputs/vec --distill --sysid --terminal_value --mlp_dim 128 --latent_dim 128 --steps 150000
  run_train "netsize_default_rep${rep}" outputs/vec --distill --sysid --terminal_value --steps 150000
  run_train "netsize_large_rep${rep}"   outputs/vec --distill --sysid --terminal_value --mlp_dim 512 --latent_dim 512 --steps 150000

  # 4. Train-without-Q
  run_train "e_no_q_rep${rep}" outputs/vec --distill --sysid --terminal_value --actor_mode bc_no_q --sim_data_ratio 0.75 --steps 150000
  run_train "f_no_q_rep${rep}" outputs/vec --distill --sysid --terminal_value --actor_mode bc_mu_no_q --elite_target mean --sim_data_ratio 0.75 --prior_coef 2.0 --steps 150000

  # 5. Offline training v1 (3 near-duplicate conditions)
  run_train "offline_v1_rep${rep}"    outputs/vec --distill --sysid --terminal_value --offline_start_val_loss 0.15 --steps 150000
  run_train "offline_v1_01_rep${rep}" outputs/vec --distill --sysid --terminal_value --offline_start_val_loss 0.1  --steps 150000
  run_train "offline_v1_02_rep${rep}" outputs/vec --distill --sysid --terminal_value --offline_start_val_loss 0.1  --steps 150000

  # 6. DAgger beta sweep
  run_train "dagger_b0_rep${rep}"  outputs/vec --distill --sysid --terminal_value --actor_mode bc_mu --dagger_beta 0.0 --steps 150000
  run_train "dagger_b30_rep${rep}" outputs/vec --distill --sysid --terminal_value --actor_mode bc_mu --dagger_beta 0.3 --steps 150000
  run_train "dagger_b60_rep${rep}" outputs/vec --distill --sysid --terminal_value --actor_mode bc_mu --dagger_beta 0.6 --steps 150000

  # 7. Training at reduced rollout budget
  run_train "dagger_b30_s128_rep${rep}" outputs/vec --distill --sysid --terminal_value --actor_mode bc_mu --dagger_beta 0.3 --num_samples 128 --steps 150000
  run_train "d_termq_s128_rep${rep}"    outputs/vec --distill --sysid --terminal_value --num_samples 128 --steps 150000

  # 8. Pure RL (no planner)
  run_train "pure_rl_v1_rep${rep}" outputs/vec --pure_rl --steps 150000
done

# 9. Offline v2 trigger sweep: top up to OFFLINE_V2_TARGET_REPS identically
#    configured samples per condition (finished r1/r2 runs are skipped).
log "--- offline_v2 sweep: top up to ${OFFLINE_V2_TARGET_REPS} samples/condition ---"
declare -A OFFLINE_V2_FLAGS=(
  [baseline]=""
  [forced_early]="--offline_force_step 20000 --offline_mode once"
  [forced_late]="--offline_force_step 45000 --offline_mode once"
  [periodic]="--offline_start_val_loss 0.15 --offline_confirm_checks 2 --offline_mode periodic --offline_periodic_num_updates 8000"
)
for cond in baseline forced_early forced_late periodic; do
  # shellcheck disable=SC2206
  extra=(${OFFLINE_V2_FLAGS[$cond]})
  for r in $(seq 1 "$OFFLINE_V2_TARGET_REPS"); do
    run_train "offline_v2_${cond}_r${r}" outputs/vec --distill --sysid --terminal_value --steps 60000 "${extra[@]}"
  done
done

log "=== sweep done === ok=$N_OK skip=$N_SKIP fail=$N_FAIL  total wall time: $(fmt_dur "$SECONDS")"
if [[ $N_FAIL -gt 0 ]]; then
  log "FAILED: ${FAILED_JOBS[*]}"
  log "rerun this script to retry; completed jobs are skipped."
  exit 1
fi
