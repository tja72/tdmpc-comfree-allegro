#!/usr/bin/env bash
# Regenerate every figure and video of the post. Run from anywhere; works in blogpost/.
# Baseline / sim2sim slots stay PENDING unless extra args are passed through, e.g.
#   BASELINE="--baseline_run ../tdmpc-square-pearl/logs/comfree-allegro-cube/0/residual_1m --max_step 1000000" \
#   F6_BASELINE="--baseline_run ../tdmpc-square-pearl/logs/comfree-allegro-cube/0/residual_1m --max_step 150000" \
#   scripts/make_all_figures.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/comfree-tdmpc/bin/python}
$PY scripts/fig_f1_architecture.py
$PY scripts/fig_f2_throughput.py
$PY scripts/fig_f3_ladder_curves.py
$PY scripts/fig_f4_long_runs.py ${BASELINE:-}
$PY scripts/fig_f5_sysid.py
$PY scripts/fig_f6_planner_vs_policy.py ${F6_BASELINE:-}
$PY scripts/fig_sim2sim.py
$PY scripts/fig_a1_policy_gap.py
$PY scripts/fig_a2_rollout_budget.py
$PY scripts/make_videos.py
echo "[make_all_figures] done"
