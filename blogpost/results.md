# Aggregated results (comfree_tdmpc)

Generated 2026-09-25 18:50 by

```bash
cd blogpost && /home/tim/miniconda3/envs/comfree-tdmpc/bin/python scripts/aggregate_results.py
```
Source of every row: `comfree_tdmpc/outputs/<dir>` (listed per row). Machine-readable copy: `blogpost/results.json`.

**Conventions.** *final* = the last evaluation row of that mode (pi = policy alone, evaluated every 5k env steps; mpc = planner, every 25k). *tail* = mean of the last 6 pi / last 3 mpc evaluations, the convention of `comfree_tdmpc/scripts/report/make_ablation_figures.py` (→ `outputs/report/figures/summary.json`, which the README table quotes). Each evaluation is 16 episodes of 120 control steps. For the 60k-step offline_v2 runs the mpc tail covers all 3 mpc evals. *G* = goals reached per episode, *drop* = fraction of episodes where the cube fell, *R* = return. Across repetitions: **mean ± sample std (ddof=1)**; n is the number of runs; `(single run)` where n=1. Repetitions are same-seed reruns on a non-deterministic GPU simulator, not independent seeds (EXPERIMENTS.md, conventions). *wall h* = `train.csv` `wall_time` of the last row; rep-1 runs of several vec configs ran on a slower/shared GPU (RTX 3070 vs 4090, EXPERIMENTS.md) so wall-clock is not comparable across reps. *extra updates* = `updates` − 1.5·(steps − 2000 seed steps): >0 means an offline phase actually ran (50000 for a full once-phase, 8000 per periodic phase).

**Degenerate runs** (EXPERIMENTS.md §8; auto-check: every pi and mpc evaluation drops all episodes, agrees = True): `ablation_300k_rep3/d_termq`, `vec/e_no_q_rep3`, `vec/offline_v2_baseline_r1`. Groups containing one are reported twice: all runs, and **excl. degenerate**.

## 1. Ablation ladder, 150k steps (`ablation/`, `ablation_rep2/`, `ablation_rep3/`)

| config | n | mpc goals/ep final | mpc goals/ep tail | mpc drop tail | pi goals/ep final | pi goals/ep tail | pi drop tail | wall h | source (outputs/) |
|---|---|---|---|---|---|---|---|---|---|
| `a_base` | 3 | 1.48 ± 0.19 | 1.44 ± 0.29 | 0.73 ± 0.04 | 0.44 ± 0.38 | 0.40 ± 0.35 | 0.86 ± 0.15 | 0.9 ± 0.1 | `ablation/a_base`, `ablation_rep2/a_base`, `ablation_rep3/a_base` |
| `b_distill` | 3 | 1.83 ± 0.31 | 1.80 ± 0.23 | 0.78 ± 0.10 | 1.02 ± 0.36 | 0.94 ± 0.17 | 0.80 ± 0.06 | 0.9 ± 0.1 | `ablation/b_distill`, `ablation_rep2/b_distill`, `ablation_rep3/b_distill` |
| `c_sysid` | 3 | 2.75 ± 0.70 | 2.56 ± 0.55 | 0.77 ± 0.13 | 1.19 ± 0.60 | 1.08 ± 0.28 | 0.70 ± 0.08 | 1.0 ± 0.1 | `ablation/c_sysid`, `ablation_rep2/c_sysid`, `ablation_rep3/c_sysid` |
| `d_termq` | 3 | 4.75 ± 0.38 | 4.72 ± 0.22 | 0.02 ± 0.00 | 0.81 ± 0.33 | 0.77 ± 0.28 | 0.47 ± 0.04 | 1.0 ± 0.1 | `ablation/d_termq`, `ablation_rep2/d_termq`, `ablation_rep3/d_termq` |
| `e_bc` | 3 | 5.10 ± 0.13 | 4.91 ± 0.37 | 0.04 ± 0.02 | 0.96 ± 0.38 | 0.80 ± 0.14 | 0.26 ± 0.05 | 1.0 ± 0.1 | `ablation/e_bc`, `ablation_rep2/e_bc`, `ablation_rep3/e_bc` |
| `f_meantarget` | 3 | 5.10 ± 0.51 | 4.57 ± 0.28 | 0.03 ± 0.01 | 1.10 ± 0.07 | 0.99 ± 0.08 | 0.20 ± 0.02 | 1.0 ± 0.1 | `ablation/f_meantarget`, `ablation_rep2/f_meantarget`, `ablation_rep3/f_meantarget` |

Pooled, identical configuration (config.json diff is empty apart from later-added inert defaults):

| config | n | mpc goals/ep final | mpc goals/ep tail | mpc drop tail | pi goals/ep final | pi goals/ep tail | pi drop tail | wall h | source (outputs/) |
|---|---|---|---|---|---|---|---|---|---|
| rung D config @150k (ablation d_termq x3 + netsize_default x3) | 6 | 4.72 ± 0.65 | 4.61 ± 0.43 | 0.02 ± 0.02 | 0.74 ± 0.23 | 0.65 ± 0.22 | 0.47 ± 0.07 | 1.5 ± 1.1 | `ablation/d_termq`, `ablation_rep2/d_termq`, `ablation_rep3/d_termq`, `vec/netsize_default`, `vec/netsize_default_rep2`, `vec/netsize_default_rep3` |
| offline_v1 threshold 0.1 (offline_v1_01 x3 + offline_v1_02 x3) | 6 | 5.08 ± 0.34 | 4.67 ± 0.23 | 0.02 ± 0.02 | 0.80 ± 0.30 | 0.78 ± 0.19 | 0.51 ± 0.02 | 4.1 ± 1.5 | `vec/offline_v1_01`, `vec/offline_v1_01_rep2`, `vec/offline_v1_01_rep3`, `vec/offline_v1_02`, `vec/offline_v1_02_rep2`, `vec/offline_v1_02_rep3` |
| offline_v1 threshold 0.1, only runs where the offline phase fired | 5 | 5.08 ± 0.38 | 4.68 ± 0.26 | 0.03 ± 0.02 | 0.82 ± 0.33 | 0.84 ± 0.15 | 0.51 ± 0.03 | 4.7 ± 0.6 | `vec/offline_v1_01`, `vec/offline_v1_01_rep2`, `vec/offline_v1_01_rep3`, `vec/offline_v1_02`, `vec/offline_v1_02_rep2` |

## 2. Rungs D and E at 300k steps (`ablation_300k{,_rep2,_rep3}/`)

| config | n | mpc goals/ep final | mpc goals/ep tail | mpc drop tail | pi goals/ep final | pi goals/ep tail | pi drop tail | wall h | source (outputs/) |
|---|---|---|---|---|---|---|---|---|---|
| `d_termq@300k` | 3 | 3.62 ± 3.14 | 3.56 ± 3.09 | 0.34 ± 0.57 | 1.17 ± 1.19 | 1.07 ± 1.04 | 0.62 ± 0.33 | 3.6 ± 2.7 | `ablation_300k/d_termq`, `ablation_300k_rep2/d_termq`, `ablation_300k_rep3/d_termq` |
| `d_termq@300k` **excl. degenerate** | 2 | 5.44 ± 0.27 | 5.34 ± 0.07 | 0.01 ± 0.01 | 1.75 ± 0.88 | 1.61 ± 0.67 | 0.43 ± 0.06 | 4.2 ± 3.4 | `ablation_300k/d_termq`, `ablation_300k_rep2/d_termq` |
| `e_bc@300k` | 3 | 5.81 ± 0.31 | 5.60 ± 0.10 | 0.00 ± 0.00 | 1.17 ± 0.69 | 1.11 ± 0.25 | 0.20 ± 0.04 | 2.9 ± 1.2 | `ablation_300k/e_bc`, `ablation_300k_rep2/e_bc`, `ablation_300k_rep3/e_bc` |

## 3. Policy-net size (`vec/netsize_*`, all with rung-D flags)

| config | n | mpc goals/ep final | mpc goals/ep tail | mpc drop tail | pi goals/ep final | pi goals/ep tail | pi drop tail | wall h | source (outputs/) |
|---|---|---|---|---|---|---|---|---|---|
| `netsize_small` | 3 | 4.62 ± 0.27 | 4.56 ± 0.12 | 0.05 ± 0.02 | 0.40 ± 0.22 | 0.39 ± 0.13 | 0.31 ± 0.08 | 2.0 ± 1.5 | `vec/netsize_small`, `vec/netsize_small_rep2`, `vec/netsize_small_rep3` |
| `netsize_default` | 3 | 4.69 ± 0.96 | 4.50 ± 0.61 | 0.03 ± 0.03 | 0.67 ± 0.10 | 0.54 ± 0.08 | 0.47 ± 0.10 | 2.0 ± 1.5 | `vec/netsize_default`, `vec/netsize_default_rep2`, `vec/netsize_default_rep3` |
| `netsize_large` | 3 | 4.33 ± 0.50 | 4.30 ± 0.24 | 0.03 ± 0.01 | 0.94 ± 0.19 | 0.76 ± 0.20 | 0.44 ± 0.04 | 1.5 ± 0.6 | `vec/netsize_large`, `vec/netsize_large_rep2`, `vec/netsize_large_rep3` |

## 4. Actor loss without Q (`vec/e_no_q*`, `vec/f_no_q*`) — compare with rungs E/F in §1

| config | n | mpc goals/ep final | mpc goals/ep tail | mpc drop tail | pi goals/ep final | pi goals/ep tail | pi drop tail | wall h | source (outputs/) |
|---|---|---|---|---|---|---|---|---|---|
| `e_no_q` | 3 | 3.42 ± 2.97 | 3.17 ± 2.76 | 0.35 ± 0.57 | 0.35 ± 0.42 | 0.35 ± 0.30 | 0.48 ± 0.45 | 1.0 ± 0.0 | `vec/e_no_q`, `vec/e_no_q_rep2`, `vec/e_no_q_rep3` |
| `e_no_q` **excl. degenerate** | 2 | 5.12 ± 0.35 | 4.76 ± 0.28 | 0.02 ± 0.03 | 0.53 ± 0.40 | 0.52 ± 0.03 | 0.22 ± 0.06 | 1.0 ± 0.1 | `vec/e_no_q`, `vec/e_no_q_rep2` |
| `f_no_q` | 3 | 4.58 ± 0.30 | 4.44 ± 0.34 | 0.02 ± 0.04 | 0.56 ± 0.44 | 0.70 ± 0.13 | 0.27 ± 0.08 | 1.4 ± 0.6 | `vec/f_no_q`, `vec/f_no_q_rep2`, `vec/f_no_q_rep3` |

## 5. DAgger and training-time planner budget (`vec/dagger_*`, `vec/d_termq_s128*`)

| config | n | mpc goals/ep final | mpc goals/ep tail | mpc drop tail | pi goals/ep final | pi goals/ep tail | pi drop tail | wall h | source (outputs/) |
|---|---|---|---|---|---|---|---|---|---|
| `dagger_b0` | 3 | 4.54 ± 0.10 | 4.53 ± 0.25 | 0.03 ± 0.03 | 0.58 ± 0.24 | 0.62 ± 0.16 | 0.26 ± 0.01 | 1.6 ± 0.9 | `vec/dagger_b0`, `vec/dagger_b0_rep2`, `vec/dagger_b0_rep3` |
| `dagger_b30` | 3 | 4.42 ± 0.24 | 4.29 ± 0.45 | 0.04 ± 0.04 | 0.85 ± 0.46 | 0.69 ± 0.09 | 0.18 ± 0.07 | 1.4 ± 0.6 | `vec/dagger_b30`, `vec/dagger_b30_rep2`, `vec/dagger_b30_rep3` |
| `dagger_b60` | 3 | 3.81 ± 0.43 | 3.79 ± 0.11 | 0.09 ± 0.06 | 0.79 ± 0.38 | 0.89 ± 0.21 | 0.18 ± 0.03 | 1.4 ± 0.6 | `vec/dagger_b60`, `vec/dagger_b60_rep2`, `vec/dagger_b60_rep3` |
| `dagger_b30_s128` | 3 | 3.75 ± 0.45 | 3.69 ± 0.22 | 0.05 ± 0.05 | 1.08 ± 0.38 | 0.88 ± 0.20 | 0.18 ± 0.02 | 0.9 ± 0.1 | `vec/dagger_b30_s128`, `vec/dagger_b30_s128_rep2`, `vec/dagger_b30_s128_rep3` |
| `d_termq_s128` | 3 | 4.56 ± 0.17 | 4.17 ± 0.15 | 0.02 ± 0.02 | 0.88 ± 0.38 | 0.77 ± 0.05 | 0.39 ± 0.06 | 0.9 ± 0.1 | `vec/d_termq_s128`, `vec/d_termq_s128_rep2`, `vec/d_termq_s128_rep3` |

## 6. Offline consolidation phase v1, 150k steps (`vec/offline_v1*`)

`offline_v1` = trigger val_loss<0.15; `offline_v1_01` and `offline_v1_02` = identical configs, trigger 0.1 (pooled n=6 in §1's pooled table).

| config | n | mpc goals/ep final | mpc goals/ep tail | mpc drop tail | pi goals/ep final | pi goals/ep tail | pi drop tail | wall h | source (outputs/) |
|---|---|---|---|---|---|---|---|---|---|
| `offline_v1` | 3 | 4.94 ± 0.22 | 4.66 ± 0.15 | 0.01 ± 0.01 | 0.73 ± 0.50 | 0.66 ± 0.23 | 0.52 ± 0.07 | 4.9 ± 0.2 | `vec/offline_v1`, `vec/offline_v1_rep2`, `vec/offline_v1_rep3` |
| `offline_v1_01` | 3 | 5.23 ± 0.44 | 4.56 ± 0.26 | 0.03 ± 0.01 | 0.96 ± 0.31 | 0.89 ± 0.15 | 0.50 ± 0.03 | 4.7 ± 0.6 | `vec/offline_v1_01`, `vec/offline_v1_01_rep2`, `vec/offline_v1_01_rep3` |
| `offline_v1_02` | 3 | 4.94 ± 0.17 | 4.78 ± 0.19 | 0.01 ± 0.02 | 0.65 ± 0.25 | 0.67 ± 0.19 | 0.51 ± 0.02 | 3.4 ± 2.1 | `vec/offline_v1_02`, `vec/offline_v1_02_rep2`, `vec/offline_v1_02_rep3` |

Did the offline phase fire? (`extra updates` from train.csv)

| run | offline config | extra updates | fired |
|---|---|---|---|
| `vec/offline_v1` | {"offline_start_val_loss": 0.15} | 50000 | yes |
| `vec/offline_v1_01` | {"offline_start_val_loss": 0.1} | 50000 | yes |
| `vec/offline_v1_01_rep2` | {"offline_start_val_loss": 0.1, "offline_mode": "once"} | 50000 | yes |
| `vec/offline_v1_01_rep3` | {"offline_start_val_loss": 0.1, "offline_mode": "once"} | 50000 | yes |
| `vec/offline_v1_02` | {"offline_start_val_loss": 0.1} | 50000 | yes |
| `vec/offline_v1_02_rep2` | {"offline_start_val_loss": 0.1, "offline_mode": "once"} | 50000 | yes |
| `vec/offline_v1_02_rep3` | {"offline_start_val_loss": 0.1, "offline_mode": "once"} | 0 | **no** |
| `vec/offline_v1_rep2` | {"offline_start_val_loss": 0.15, "offline_mode": "once"} | 50000 | yes |
| `vec/offline_v1_rep3` | {"offline_start_val_loss": 0.15, "offline_mode": "once"} | 50000 | yes |
| `vec/offline_v2_baseline_r1` | {"offline_mode": "once"} | 0 | **no** |
| `vec/offline_v2_baseline_r2` | {"offline_mode": "once"} | 0 | **no** |
| `vec/offline_v2_baseline_r3` | {"offline_mode": "once"} | 0 | **no** |
| `vec/offline_v2_forced_early_r1` | {"offline_force_step": 20000, "offline_mode": "once"} | 50000 | yes |
| `vec/offline_v2_forced_early_r2` | {"offline_force_step": 20000, "offline_mode": "once"} | 50000 | yes |
| `vec/offline_v2_forced_early_r3` | {"offline_force_step": 20000, "offline_mode": "once"} | 50000 | yes |
| `vec/offline_v2_forced_late_r1` | {"offline_force_step": 45000, "offline_mode": "once"} | 50000 | yes |
| `vec/offline_v2_forced_late_r2` | {"offline_force_step": 45000, "offline_mode": "once"} | 50000 | yes |
| `vec/offline_v2_forced_late_r3` | {"offline_force_step": 45000, "offline_mode": "once"} | 50000 | yes |
| `vec/offline_v2_periodic_r1` | {"offline_start_val_loss": 0.15, "offline_mode": "periodic"} | 8000 | yes |
| `vec/offline_v2_periodic_r2` | {"offline_start_val_loss": 0.15, "offline_mode": "periodic"} | 0 | **no** |
| `vec/offline_v2_periodic_r3` | {"offline_start_val_loss": 0.15, "offline_mode": "periodic"} | 0 | **no** |

## 7. Offline phase v2 trigger sweep, 60k steps (`vec/offline_v2_*_r{1,2,3}`)

| config | n | mpc goals/ep final | mpc goals/ep tail | mpc drop tail | pi goals/ep final | pi goals/ep tail | pi drop tail | wall h | source (outputs/) |
|---|---|---|---|---|---|---|---|---|---|
| `offline_v2_baseline` | 3 | 2.42 ± 2.14 | 2.27 ± 2.00 | 0.44 ± 0.49 | 0.33 ± 0.34 | 0.33 ± 0.31 | 0.70 ± 0.27 | 0.4 ± 0.0 | `vec/offline_v2_baseline_r1`, `vec/offline_v2_baseline_r2`, `vec/offline_v2_baseline_r3` |
| `offline_v2_baseline` **excl. degenerate** | 2 | 3.62 ± 0.62 | 3.41 ± 0.52 | 0.17 ± 0.12 | 0.50 ± 0.27 | 0.49 ± 0.19 | 0.55 ± 0.09 | 0.4 ± 0.0 | `vec/offline_v2_baseline_r2`, `vec/offline_v2_baseline_r3` |
| `offline_v2_forced_early` | 3 | 3.65 ± 0.28 | 3.40 ± 0.35 | 0.17 ± 0.07 | 0.33 ± 0.10 | 0.26 ± 0.11 | 0.58 ± 0.07 | 4.0 ± 0.5 | `vec/offline_v2_forced_early_r1`, `vec/offline_v2_forced_early_r2`, `vec/offline_v2_forced_early_r3` |
| `offline_v2_forced_late` | 3 | 3.73 ± 0.36 | 3.58 ± 0.08 | 0.11 ± 0.09 | 0.33 ± 0.20 | 0.37 ± 0.20 | 0.49 ± 0.04 | 4.1 ± 0.6 | `vec/offline_v2_forced_late_r1`, `vec/offline_v2_forced_late_r2`, `vec/offline_v2_forced_late_r3` |
| `offline_v2_periodic` | 3 | 4.25 ± 0.27 | 3.62 ± 0.18 | 0.06 ± 0.04 | 0.44 ± 0.11 | 0.50 ± 0.29 | 0.41 ± 0.04 | 0.6 ± 0.4 | `vec/offline_v2_periodic_r1`, `vec/offline_v2_periodic_r2`, `vec/offline_v2_periodic_r3` |

## 8. Pure RL, no planner (`vec/pure_rl_v1*`)

| config | n | mpc goals/ep final | mpc goals/ep tail | mpc drop tail | pi goals/ep final | pi goals/ep tail | pi drop tail | wall h | source (outputs/) |
|---|---|---|---|---|---|---|---|---|---|
| `pure_rl_v1` | 3 | NA | NA | NA | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.3 ± 0.0 | `vec/pure_rl_v1`, `vec/pure_rl_v1_rep2`, `vec/pure_rl_v1_rep3` |

(no mpc columns: the planner is never used, so mpc is NA.)

## 9. TD-MPC2 baseline (pearl, `comfree-allegro-cube`)

**PENDING** — no finished baseline run exists; the partial `residual_1m` run (tdmpc-square-pearl/logs/comfree-allegro-cube/0/residual_1m) is not a result (the run was still in progress when this table was generated). Fill with `--baseline_run <pearl run dir> [--baseline_run ...] --max_step 150000 300000 1000000`.

| budget | mpc G final | mpc G tail | mpc drop tail | pi G final | pi G tail | n seeds | source |
|---|---|---|---|---|---|---|---|
| 150k (= ladder budget) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 300k (= 300k runs / hero) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 1M (full run) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## 10. Rollout-budget sweep, eval only (`outputs/rollout_budget/results.json`)

One checkpoint each (rep 1 of `vec/dagger_b30` and `ablation/d_termq`), 32 episodes per cell, seed 77, `num_pi_trajs`=24.

| iters x samples (budget) | dagger_b30 G | dagger_b30 drop | dagger_b30 ms/plan | d_termq G | d_termq drop | d_termq ms/plan |
|---|---|---|---|---|---|---|
| 3x256 (768) | 5.19 | 0.03 | 226 | 4.69 | 0.06 | 221 |
| 3x128 (384) | 4.81 | 0.00 | 146 | 4.34 | 0.16 | 143 |
| 3x64 (192) | 4.38 | 0.00 | 111 | 4.16 | 0.00 | 110 |
| 2x64 (128) | 4.12 | 0.00 | 75 | 3.91 | 0.03 | 75 |
| 1x64 (64) | 4.03 | 0.00 | 42 | 3.44 | 0.03 | 42 |
| 1x32 (32) | 2.44 | 0.06 | 37 | 2.16 | 0.09 | 37 |

All repetitions pooled (reps [1, 2, 3]: rep k = `vec/dagger_b30[_repk]` vs `ablation[_repk]/d_termq`; reps 2/3 in `outputs/rollout_budget_rep{2,3}/results.json`), mean ± sample std over checkpoints, n per cell:

| iters x samples (budget) | dagger_b30 G | drop | ms/plan | d_termq G | drop | ms/plan | n (dagger/d_termq) | reps where dagger > d_termq |
|---|---|---|---|---|---|---|---|---|
| 3x256 (768) | 4.64 ± 0.52 | 0.02 | 224 | 4.76 ± 0.33 | 0.02 | 221 | 3/3 | 1/3 |
| 3x128 (384) | 4.28 ± 0.53 | 0.02 | 145 | 4.61 ± 0.34 | 0.06 | 143 | 3/3 | 1/3 |
| 3x64 (192) | 4.00 ± 0.36 | 0.02 | 110 | 4.18 ± 0.31 | 0.01 | 110 | 3/3 | 1/3 |
| 2x64 (128) | 3.74 ± 0.41 | 0.00 | 75 | 3.85 ± 0.36 | 0.04 | 75 | 3/3 | 1/3 |
| 1x64 (64) | 3.39 ± 0.56 | 0.01 | 42 | 3.55 ± 0.28 | 0.01 | 42 | 3/3 | 1/3 |
| 1x32 (32) | 2.21 ± 0.25 | 0.05 | 37 | 2.34 ± 0.50 | 0.05 | 37 | 3/3 | 1/3 |

## 11. Benchmarks (`06_throughput/`, `07_verify/`, `08_tune/`)

- `06_throughput/throughput.json`: 92.7k control steps/s (0.93M physics steps/s) at 512 worlds; peak 1.80M physics steps/s at 4096 worlds; 1 world: 377 control steps/s. Plan call: 256 worlds 331 ms (learned-net overhead 4.4 ms); 1024 worlds 492 ms (learned-net overhead 15.3 ms). Update: batch 256 7.4 ms; batch 512 7.5 ms; batch 1024 8.2 ms.
- `07_verify/vec.json`: all 6 correctness checks ok=True (vec env vs single env after 1 step: 3.0e-08); speedup at 16 envs 4.19x (25.9 env steps/s), at 32 envs 4.32x.
- `08_tune/planner.json` (single-env MPPI, no learning — planner alone drops the cube in most episodes):

| config | H | iters | samples | goals/1k steps | drop rate | goals/GPU-min | episodes |
|---|---|---|---|---|---|---|---|
| base | 8 | 4 | 256 | 39.0 | 0.95 | 113.3 | 58 |
| iterations=1 | 8 | 1 | 256 | 25.6 | 0.98 | 223.3 | 55 |
| iterations=2 | 8 | 2 | 256 | 28.3 | 0.98 | 146.1 | 59 |
| iterations=3 | 8 | 3 | 256 | 36.6 | 0.95 | 138.1 | 56 |
| num_samples=128 | 8 | 4 | 128 | 33.1 | 0.93 | 148.8 | 55 |
| num_samples=512 | 8 | 4 | 512 | 35.7 | 1.00 | 58.9 | 61 |
| horizon=4 | 4 | 4 | 256 | 23.6 | 0.97 | 132.6 | 76 |
| horizon=6 | 6 | 4 | 256 | 31.7 | 1.00 | 127.7 | 82 |
| horizon=12 | 12 | 4 | 256 | 33.2 | 0.84 | 61.6 | 43 |

## 12. SysID error, first vs last SysID call (`<run>/sysid.csv`, mean |u_hat − u*| per group)

Per configuration, mean ± sample std over reps of first → last:

| config | n | joint | actuator | contact | inertial | comfree | param_err_mean |
|---|---|---|---|---|---|---|---|
| `c_sysid` | 3 | 0.085→0.082 (±0.004) | 0.103→0.095 (±0.006) | 0.150→0.150 (±0.021) | 0.085→0.118 (±0.010) | 0.183→0.081 (±0.010) | 0.096→0.093 (±0.005) |
| `d_termq` | 3 | 0.084→0.080 (±0.005) | 0.100→0.083 (±0.013) | 0.151→0.128 (±0.035) | 0.089→0.100 (±0.022) | 0.184→0.083 (±0.025) | 0.096→0.087 (±0.003) |
| `e_bc` | 3 | 0.085→0.085 (±0.015) | 0.102→0.064 (±0.029) | 0.153→0.141 (±0.041) | 0.094→0.113 (±0.028) | 0.199→0.082 (±0.007) | 0.098→0.090 (±0.017) |
| `f_meantarget` | 3 | 0.086→0.089 (±0.003) | 0.104→0.082 (±0.003) | 0.153→0.122 (±0.033) | 0.089→0.107 (±0.030) | 0.191→0.115 (±0.027) | 0.098→0.094 (±0.003) |
| `d_termq@300k` | 3 | 0.085→0.091 (±0.007) | 0.102→0.087 (±0.020) | 0.149→0.113 (±0.052) | 0.087→0.120 (±0.026) | 0.193→0.110 (±0.018) | 0.097→0.096 (±0.010) |
| `e_bc@300k` | 3 | 0.087→0.086 (±0.003) | 0.103→0.094 (±0.035) | 0.155→0.053 (±0.020) | 0.086→0.100 (±0.028) | 0.198→0.102 (±0.016) | 0.099→0.085 (±0.006) |
| `d_termq_s128` | 3 | 0.085→0.089 (±0.015) | 0.105→0.090 (±0.015) | 0.151→0.159 (±0.023) | 0.086→0.114 (±0.018) | 0.187→0.090 (±0.067) | 0.097→0.098 (±0.010) |
| `dagger_b0` | 3 | 0.087→0.087 (±0.003) | 0.102→0.081 (±0.041) | 0.149→0.126 (±0.021) | 0.088→0.118 (±0.018) | 0.191→0.086 (±0.032) | 0.097→0.093 (±0.008) |
| `dagger_b30` | 3 | 0.084→0.086 (±0.003) | 0.100→0.084 (±0.020) | 0.154→0.140 (±0.030) | 0.089→0.094 (±0.009) | 0.186→0.059 (±0.014) | 0.096→0.091 (±0.003) |
| `dagger_b30_s128` | 3 | 0.085→0.093 (±0.009) | 0.103→0.069 (±0.031) | 0.154→0.167 (±0.021) | 0.089→0.129 (±0.006) | 0.196→0.098 (±0.035) | 0.098→0.101 (±0.005) |
| `dagger_b60` | 3 | 0.083→0.088 (±0.005) | 0.099→0.092 (±0.023) | 0.153→0.161 (±0.009) | 0.087→0.106 (±0.015) | 0.183→0.075 (±0.033) | 0.095→0.097 (±0.003) |
| `e_no_q` | 3 | 0.086→0.083 (±0.007) | 0.104→0.091 (±0.019) | 0.149→0.138 (±0.054) | 0.091→0.118 (±0.023) | 0.189→0.084 (±0.040) | 0.098→0.092 (±0.010) |
| `f_no_q` | 3 | 0.085→0.093 (±0.007) | 0.103→0.096 (±0.020) | 0.147→0.133 (±0.015) | 0.093→0.143 (±0.034) | 0.192→0.065 (±0.033) | 0.097→0.101 (±0.004) |
| `netsize_default` | 3 | 0.085→0.078 (±0.001) | 0.105→0.104 (±0.020) | 0.148→0.110 (±0.037) | 0.090→0.113 (±0.006) | 0.189→0.082 (±0.025) | 0.097→0.087 (±0.006) |
| `netsize_large` | 3 | 0.085→0.089 (±0.011) | 0.099→0.086 (±0.019) | 0.150→0.147 (±0.041) | 0.087→0.136 (±0.018) | 0.193→0.088 (±0.045) | 0.097→0.099 (±0.009) |
| `netsize_small` | 3 | 0.085→0.093 (±0.004) | 0.102→0.110 (±0.004) | 0.149→0.146 (±0.029) | 0.088→0.130 (±0.020) | 0.194→0.131 (±0.025) | 0.097→0.104 (±0.005) |
| `offline_v1` | 3 | 0.085→0.085 (±0.008) | 0.100→0.081 (±0.010) | 0.146→0.127 (±0.012) | 0.084→0.120 (±0.012) | 0.188→0.104 (±0.043) | 0.096→0.092 (±0.002) |
| `offline_v1_01` | 3 | 0.084→0.084 (±0.002) | 0.102→0.080 (±0.027) | 0.147→0.138 (±0.023) | 0.088→0.098 (±0.009) | 0.184→0.101 (±0.022) | 0.096→0.091 (±0.002) |
| `offline_v1_02` | 3 | 0.085→0.083 (±0.005) | 0.099→0.087 (±0.018) | 0.148→0.125 (±0.009) | 0.090→0.138 (±0.026) | 0.192→0.109 (±0.033) | 0.096→0.093 (±0.005) |
| `offline_v2_baseline` | 3 | 0.085→0.084 (±0.006) | 0.108→0.093 (±0.014) | 0.148→0.157 (±0.011) | 0.088→0.105 (±0.037) | 0.185→0.185 (±0.025) | 0.097→0.097 (±0.005) |
| `offline_v2_forced_early` | 3 | 0.085→0.084 (±0.005) | 0.101→0.081 (±0.008) | 0.150→0.145 (±0.022) | 0.085→0.093 (±0.001) | 0.189→0.189 (±0.010) | 0.096→0.094 (±0.006) |
| `offline_v2_forced_late` | 3 | 0.084→0.084 (±0.001) | 0.104→0.077 (±0.015) | 0.146→0.127 (±0.014) | 0.087→0.108 (±0.008) | 0.188→0.149 (±0.017) | 0.096→0.091 (±0.003) |
| `offline_v2_periodic` | 3 | 0.085→0.084 (±0.002) | 0.101→0.075 (±0.011) | 0.151→0.149 (±0.008) | 0.085→0.107 (±0.006) | 0.187→0.169 (±0.044) | 0.096→0.094 (±0.001) |

Per run (first → last):

| run | calls | joint | actuator | contact | inertial | comfree | param_err_mean |
|---|---|---|---|---|---|---|---|
| `ablation/c_sysid` | 29 | 0.085→0.079 | 0.104→0.099 | 0.154→0.129 | 0.092→0.107 | 0.188→0.092 | 0.098→0.089 |
| `ablation/d_termq` | 29 | 0.084→0.078 | 0.097→0.072 | 0.150→0.107 | 0.095→0.106 | 0.186→0.112 | 0.096→0.084 |
| `ablation/e_bc` | 29 | 0.085→0.101 | 0.107→0.089 | 0.151→0.183 | 0.096→0.139 | 0.191→0.074 | 0.098→0.110 |
| `ablation/f_meantarget` | 29 | 0.088→0.090 | 0.101→0.082 | 0.155→0.146 | 0.089→0.090 | 0.194→0.135 | 0.099→0.096 |
| `ablation_300k/d_termq` | 59 | 0.086→0.083 | 0.099→0.111 | 0.150→0.122 | 0.090→0.115 | 0.209→0.094 | 0.097→0.093 |
| `ablation_300k/e_bc` | 59 | 0.088→0.089 | 0.101→0.070 | 0.154→0.034 | 0.086→0.068 | 0.213→0.086 | 0.099→0.080 |
| `ablation_300k_rep2/d_termq` | 59 | 0.085→0.092 | 0.102→0.074 | 0.153→0.057 | 0.083→0.097 | 0.188→0.106 | 0.097→0.087 |
| `ablation_300k_rep2/e_bc` | 59 | 0.087→0.085 | 0.109→0.134 | 0.161→0.073 | 0.085→0.112 | 0.183→0.100 | 0.099→0.092 |
| `ablation_300k_rep3/d_termq` | 59 | 0.084→0.097 | 0.104→0.077 | 0.144→0.160 | 0.087→0.148 | 0.183→0.129 | 0.095→0.106 |
| `ablation_300k_rep3/e_bc` | 59 | 0.087→0.083 | 0.099→0.078 | 0.149→0.052 | 0.088→0.121 | 0.199→0.118 | 0.098→0.084 |
| `ablation_rep2/c_sysid` | 29 | 0.085→0.086 | 0.099→0.098 | 0.150→0.171 | 0.079→0.123 | 0.176→0.075 | 0.095→0.099 |
| `ablation_rep2/d_termq` | 29 | 0.084→0.086 | 0.103→0.097 | 0.151→0.109 | 0.087→0.075 | 0.186→0.067 | 0.096→0.088 |
| `ablation_rep2/e_bc` | 29 | 0.085→0.079 | 0.097→0.072 | 0.152→0.102 | 0.094→0.083 | 0.192→0.084 | 0.096→0.081 |
| `ablation_rep2/f_meantarget` | 29 | 0.085→0.092 | 0.109→0.079 | 0.152→0.084 | 0.089→0.141 | 0.195→0.125 | 0.098→0.095 |
| `ablation_rep3/c_sysid` | 29 | 0.084→0.082 | 0.106→0.088 | 0.146→0.150 | 0.085→0.124 | 0.185→0.076 | 0.096→0.093 |
| `ablation_rep3/d_termq` | 29 | 0.085→0.076 | 0.101→0.078 | 0.153→0.168 | 0.084→0.118 | 0.179→0.069 | 0.096→0.089 |
| `ablation_rep3/e_bc` | 29 | 0.086→0.074 | 0.103→0.033 | 0.158→0.138 | 0.092→0.116 | 0.214→0.089 | 0.099→0.080 |
| `ablation_rep3/f_meantarget` | 29 | 0.086→0.086 | 0.103→0.085 | 0.152→0.136 | 0.090→0.088 | 0.183→0.084 | 0.098→0.091 |
| `vec/d_termq_s128` | 29 | 0.087→0.085 | 0.104→0.085 | 0.145→0.182 | 0.087→0.135 | 0.194→0.166 | 0.097→0.101 |
| `vec/d_termq_s128_rep2` | 29 | 0.085→0.106 | 0.104→0.107 | 0.155→0.135 | 0.088→0.103 | 0.188→0.063 | 0.097→0.108 |
| `vec/d_termq_s128_rep3` | 29 | 0.085→0.078 | 0.106→0.078 | 0.153→0.159 | 0.085→0.106 | 0.179→0.041 | 0.097→0.087 |
| `vec/dagger_b0` | 29 | 0.087→0.084 | 0.095→0.074 | 0.146→0.115 | 0.089→0.130 | 0.186→0.109 | 0.097→0.091 |
| `vec/dagger_b0_rep2` | 29 | 0.087→0.087 | 0.105→0.126 | 0.149→0.150 | 0.087→0.127 | 0.210→0.099 | 0.099→0.101 |
| `vec/dagger_b0_rep3` | 29 | 0.085→0.090 | 0.106→0.045 | 0.151→0.113 | 0.087→0.097 | 0.177→0.050 | 0.097→0.087 |
| `vec/dagger_b30` | 29 | 0.084→0.083 | 0.109→0.090 | 0.155→0.120 | 0.093→0.087 | 0.190→0.072 | 0.098→0.088 |
| `vec/dagger_b30_rep2` | 29 | 0.083→0.085 | 0.097→0.100 | 0.153→0.126 | 0.088→0.103 | 0.184→0.044 | 0.095→0.091 |
| `vec/dagger_b30_rep3` | 29 | 0.084→0.089 | 0.095→0.062 | 0.153→0.174 | 0.086→0.091 | 0.184→0.060 | 0.095→0.094 |
| `vec/dagger_b30_s128` | 29 | 0.085→0.085 | 0.107→0.102 | 0.152→0.154 | 0.090→0.131 | 0.195→0.058 | 0.097→0.097 |
| `vec/dagger_b30_s128_rep2` | 29 | 0.086→0.102 | 0.101→0.042 | 0.156→0.191 | 0.089→0.134 | 0.205→0.116 | 0.098→0.107 |
| `vec/dagger_b30_s128_rep3` | 29 | 0.086→0.093 | 0.101→0.062 | 0.156→0.156 | 0.090→0.122 | 0.188→0.121 | 0.097→0.099 |
| `vec/dagger_b60` | 29 | 0.083→0.083 | 0.101→0.119 | 0.149→0.152 | 0.087→0.090 | 0.176→0.065 | 0.094→0.094 |
| `vec/dagger_b60_rep2` | 29 | 0.083→0.091 | 0.098→0.082 | 0.159→0.159 | 0.090→0.120 | 0.186→0.047 | 0.096→0.098 |
| `vec/dagger_b60_rep3` | 29 | 0.083→0.091 | 0.096→0.076 | 0.150→0.171 | 0.085→0.108 | 0.186→0.111 | 0.094→0.099 |
| `vec/e_no_q` | 29 | 0.086→0.076 | 0.107→0.090 | 0.149→0.100 | 0.091→0.140 | 0.196→0.038 | 0.098→0.084 |
| `vec/e_no_q_rep2` | 29 | 0.085→0.084 | 0.099→0.072 | 0.145→0.114 | 0.087→0.120 | 0.190→0.103 | 0.096→0.089 |
| `vec/e_no_q_rep3` | 29 | 0.087→0.090 | 0.105→0.111 | 0.151→0.200 | 0.095→0.094 | 0.182→0.111 | 0.099→0.104 |
| `vec/f_no_q` | 29 | 0.089→0.090 | 0.103→0.109 | 0.151→0.148 | 0.099→0.181 | 0.210→0.038 | 0.101→0.104 |
| `vec/f_no_q_rep2` | 29 | 0.084→0.089 | 0.101→0.105 | 0.147→0.118 | 0.089→0.133 | 0.183→0.054 | 0.095→0.096 |
| `vec/f_no_q_rep3` | 29 | 0.083→0.102 | 0.106→0.073 | 0.144→0.132 | 0.092→0.115 | 0.183→0.102 | 0.095→0.103 |
| `vec/netsize_default` | 29 | 0.087→0.078 | 0.105→0.081 | 0.148→0.068 | 0.090→0.119 | 0.201→0.067 | 0.098→0.080 |
| `vec/netsize_default_rep2` | 29 | 0.085→0.077 | 0.107→0.112 | 0.144→0.129 | 0.089→0.108 | 0.178→0.070 | 0.096→0.088 |
| `vec/netsize_default_rep3` | 29 | 0.084→0.079 | 0.103→0.119 | 0.153→0.134 | 0.092→0.112 | 0.188→0.111 | 0.097→0.092 |
| `vec/netsize_large` | 29 | 0.086→0.101 | 0.096→0.095 | 0.156→0.191 | 0.092→0.116 | 0.187→0.046 | 0.097→0.109 |
| `vec/netsize_large_rep2` | 29 | 0.086→0.081 | 0.104→0.065 | 0.145→0.140 | 0.082→0.149 | 0.193→0.136 | 0.097→0.092 |
| `vec/netsize_large_rep3` | 29 | 0.085→0.086 | 0.097→0.099 | 0.149→0.111 | 0.086→0.142 | 0.200→0.083 | 0.096→0.095 |
| `vec/netsize_small` | 29 | 0.085→0.093 | 0.101→0.114 | 0.153→0.179 | 0.090→0.129 | 0.204→0.116 | 0.097→0.107 |
| `vec/netsize_small_rep2` | 29 | 0.087→0.097 | 0.103→0.107 | 0.148→0.124 | 0.088→0.151 | 0.199→0.159 | 0.098→0.107 |
| `vec/netsize_small_rep3` | 29 | 0.084→0.089 | 0.103→0.109 | 0.146→0.136 | 0.086→0.110 | 0.179→0.117 | 0.095→0.098 |
| `vec/offline_v1` | 29 | 0.085→0.082 | 0.107→0.092 | 0.139→0.123 | 0.087→0.113 | 0.184→0.135 | 0.096→0.091 |
| `vec/offline_v1_01` | 29 | 0.085→0.087 | 0.103→0.109 | 0.145→0.119 | 0.086→0.087 | 0.178→0.082 | 0.095→0.092 |
| `vec/offline_v1_01_rep2` | 29 | 0.084→0.083 | 0.101→0.057 | 0.149→0.163 | 0.090→0.104 | 0.187→0.095 | 0.096→0.090 |
| `vec/offline_v1_01_rep3` | 29 | 0.084→0.083 | 0.103→0.072 | 0.148→0.131 | 0.088→0.103 | 0.189→0.125 | 0.096→0.089 |
| `vec/offline_v1_02` | 29 | 0.084→0.080 | 0.105→0.088 | 0.142→0.117 | 0.094→0.168 | 0.183→0.146 | 0.096→0.094 |
| `vec/offline_v1_02_rep2` | 29 | 0.086→0.081 | 0.097→0.069 | 0.153→0.135 | 0.090→0.121 | 0.199→0.086 | 0.097→0.088 |
| `vec/offline_v1_02_rep3` | 29 | 0.084→0.090 | 0.096→0.105 | 0.149→0.123 | 0.086→0.124 | 0.196→0.094 | 0.095→0.098 |
| `vec/offline_v1_rep2` | 29 | 0.084→0.080 | 0.099→0.073 | 0.147→0.140 | 0.089→0.134 | 0.189→0.122 | 0.095→0.091 |
| `vec/offline_v1_rep3` | 29 | 0.087→0.094 | 0.095→0.076 | 0.151→0.118 | 0.075→0.114 | 0.190→0.054 | 0.096→0.095 |
| `vec/offline_v2_baseline_r1` | 11 | 0.084→0.077 | 0.107→0.077 | 0.146→0.152 | 0.092→0.146 | 0.186→0.159 | 0.097→0.093 |
| `vec/offline_v2_baseline_r2` | 11 | 0.086→0.086 | 0.104→0.097 | 0.147→0.148 | 0.082→0.073 | 0.181→0.188 | 0.097→0.095 |
| `vec/offline_v2_baseline_r3` | 11 | 0.086→0.088 | 0.113→0.104 | 0.150→0.170 | 0.089→0.096 | 0.189→0.209 | 0.098→0.102 |
| `vec/offline_v2_forced_early_r1` | 11 | 0.084→0.079 | 0.103→0.076 | 0.148→0.121 | 0.086→0.094 | 0.183→0.186 | 0.096→0.087 |
| `vec/offline_v2_forced_early_r2` | 11 | 0.087→0.088 | 0.103→0.077 | 0.160→0.152 | 0.082→0.092 | 0.193→0.200 | 0.099→0.097 |
| `vec/offline_v2_forced_early_r3` | 11 | 0.083→0.085 | 0.096→0.091 | 0.142→0.163 | 0.088→0.094 | 0.190→0.180 | 0.094→0.097 |
| `vec/offline_v2_forced_late_r1` | 11 | 0.084→0.083 | 0.103→0.064 | 0.143→0.111 | 0.087→0.100 | 0.186→0.168 | 0.095→0.087 |
| `vec/offline_v2_forced_late_r2` | 11 | 0.085→0.085 | 0.106→0.075 | 0.145→0.136 | 0.085→0.108 | 0.184→0.145 | 0.096→0.093 |
| `vec/offline_v2_forced_late_r3` | 11 | 0.083→0.084 | 0.102→0.094 | 0.149→0.135 | 0.089→0.116 | 0.194→0.135 | 0.095→0.094 |
| `vec/offline_v2_periodic_r1` | 11 | 0.085→0.085 | 0.102→0.068 | 0.148→0.156 | 0.087→0.100 | 0.184→0.219 | 0.096→0.095 |
| `vec/offline_v2_periodic_r2` | 11 | 0.085→0.081 | 0.103→0.088 | 0.151→0.151 | 0.083→0.110 | 0.186→0.153 | 0.096→0.093 |
| `vec/offline_v2_periodic_r3` | 11 | 0.084→0.086 | 0.097→0.070 | 0.154→0.140 | 0.085→0.111 | 0.191→0.135 | 0.096→0.093 |

## 13. Contradiction check: claims in EXPERIMENTS.md / README.md / earlier project notes vs multi-rep data

| claim (source) | claimed | multi-rep data | verdict |
|---|---|---|---|
| Terminal value is the largest effect: C→D drop 0.88→0.02, goals/ep more than doubled (README; project notes: 1.62→4.56) | tail, rep 1 | mpc G tail C 2.56 ± 0.55 (n=3) → D 4.72 ± 0.22 (n=3); mpc drop tail C 0.77 ± 0.13 (n=3) → D 0.02 ± 0.00 (n=3); A→D 1.44→4.72 | **Holds (n=3)** as the largest single step and for the drop rate. 'More than doubles' holds vs A (3.3x) but only 1.8x vs C on the 3-rep tail mean: C reps 2/3 reach 3.12/3.19 goals/ep in the final eval, rep 1 was a low draw. |
| Distillation helps MPPI: A→B 1.62→2.02 goals/ep (project notes) | tail, rep 1 | A 1.44 ± 0.29 (n=3) → B 1.80 ± 0.23 (n=3) | Direction holds on the mean, but the gap is about one std; weak evidence with n=3. |
| SysID (C) vs B: C's planner no better than B (README table 1.94 vs 2.02) | tail, rep 1 | B 1.80 ± 0.23 (n=3) → C 2.56 ± 0.55 (n=3) | **Rep-1-specific.** Over 3 reps C is above B on the mean (large spread). |
| BC (E) no real gain over D (project notes, policy 0.65±0.39) | tail, rep 1 | policy tail D 0.77 ± 0.28 (n=3), E 0.80 ± 0.14 (n=3); mpc tail D 4.72 ± 0.22 (n=3), E 4.91 ± 0.37 (n=3) | Holds: differences are inside the rep spread. |
| The policy does not catch up with the planner (README, project notes) | all rungs | pi/mpc tail goals ratio across rungs (3-rep means): 0.16–0.53; best policy tail of any 150k config ≤ 1.08 | **Holds (n=3 everywhere).** Drift-ratio numbers (1.05…2.35) come from single-run `diagnosis.json` (rep 1 only) and cannot be checked over reps. |
| d_termq 300k planner 5.62 goals/ep (EXPERIMENTS §3) | 5.62 final, rep 1 | final: 3.62 ± 3.14 (n=3); excl. degenerate rep3: 5.44 ± 0.27 (n=2); tail excl.: 5.34 ± 0.07 (n=2) | Rep-1 value reproduced; holds with n=2 non-degenerate reps (rep 3 degenerate, 0 goals). Longer training helps the planner vs 150k rung D. |
| e_bc 300k policy 0.50 goals/ep (EXPERIMENTS §3) | 0.50 final, rep 1 | pi final 1.17 ± 0.69 (n=3); pi tail 1.11 ± 0.25 (n=3); mpc final 5.81 ± 0.31 (n=3) | **Rep 1 is the low outlier**; the other reps' policy is 1.1–1.9. Don't quote 0.50 as the 300k result. |
| Net size: policy 0.19 → 0.75 → 0.75 (small/default/large), monotonic secondary lever; planner unaffected (EXPERIMENTS §4, project notes) | final, rep 1 | pi final small 0.40 ± 0.22 (n=3), default 0.67 ± 0.10 (n=3), large 0.94 ± 0.19 (n=3); pi tail small 0.39 ± 0.13 (n=3), default 0.54 ± 0.08 (n=3), large 0.76 ± 0.20 (n=3); mpc tail small 4.56 ± 0.12 (n=3), default 4.50 ± 0.61 (n=3), large 4.30 ± 0.24 (n=3) | Ordering small<default<large on 3-rep means: final holds, tail holds; but reps overlap (std ≈ gaps), so 'monotonic' is suggestive, not established. Planner unaffected: holds. Drift ratios single-run only. |
| netsize_default planner 5.75 (project notes) | final, rep 1 | netsize_default mpc final 4.69 ± 0.96 (n=3); rung-D config pooled n=6 final 4.72 ± 0.65 (n=6), tail 4.61 ± 0.43 (n=6) | 5.75 is the top draw of 6 identical-config runs; the config's typical planner value is the pooled mean. |
| e_no_q: result blank (EXPERIMENTS §4); 'did not finish, CUDA OOM at 112000, no config.json, excluded' (project notes) | — | On disk: 3 complete 150k runs with config.json (`vec/e_no_q.log` ends with `[trainer] done: 150000 env steps`). rep1 mpc final 5.38, pi final 0.25; all: mpc tail 3.17 ± 2.76 (n=3), pi tail 0.35 ± 0.30 (n=3); excl. degenerate rep3: mpc tail 4.76 ± 0.28 (n=2), pi tail 0.52 ± 0.03 (n=2). Compare E: pi tail 0.80 ± 0.14 (n=3) | **The notes are stale**: the crashed run was evidently rerun to completion. Result: dropping Q from the BC actor loss does not help the policy (lower than E), planner similar; rep 3 degenerate. |
| f_no_q worse than F for the policy (0.875 vs 1.06), 'no improvement over F' (project notes, EXPERIMENTS) | final, rep 1 | pi final F 1.10 ± 0.07 (n=3) vs f_no_q 0.56 ± 0.44 (n=3); pi tail F 0.99 ± 0.08 (n=3) vs 0.70 ± 0.13 (n=3) | **Holds (n=3)**, and more clearly than in rep 1. |
| dagger_b30 policy 1.38 goals/ep, 'strongest policy-catching-up result' (EXPERIMENTS §4, project notes) | final, rep 1 | pi final β0 0.58 ± 0.24 (n=3), β30 0.85 ± 0.46 (n=3), β60 0.79 ± 0.38 (n=3); pi tail β0 0.62 ± 0.16 (n=3), β30 0.69 ± 0.09 (n=3), β60 0.89 ± 0.21 (n=3) | **Does not hold with n=3.** 1.38 is rep 1 only (reps 2/3: 0.50, 0.69); no β is distinguishable from the others on the policy. Planner cost: mpc tail β0 4.53 ± 0.25 (n=3), β30 4.29 ± 0.45 (n=3), β60 3.79 ± 0.11 (n=3) — higher β lowers the planner. |
| Trained at 128 samples, default beats DAgger: d_termq_s128 pi 1.25 / mpc 4.44 vs dagger_b30_s128 0.69 / 3.25 (project notes) | final, rep 1 | pi final d_termq_s128 0.88 ± 0.38 (n=3) vs dagger_b30_s128 1.08 ± 0.38 (n=3); mpc final 4.56 ± 0.17 (n=3) vs 3.75 ± 0.45 (n=3); mpc tail 4.17 ± 0.15 (n=3) vs 3.69 ± 0.22 (n=3) | **Partly.** Planner side holds (n=3). Policy side reverses on the mean: DAgger's s128 policy is not worse over 3 reps. |
| Rollout budget: dagger_b30 at 3x128 reaches 4.81 > d_termq full-budget 4.69 (EXPERIMENTS §7, project notes) | eval-only, 32 ep | 4.81 vs 4.69 (146 vs 221 ms); dagger_b30 above d_termq in all 6/6 cells | **Does not hold with n=3.** Rep 1 reproduces, but reps 2/3 reverse it: on the 3-rep mean d_termq is at or above dagger_b30 at every budget, so DAgger does not buy cheaper planning. What holds for both arms: planning degrades gracefully with fewer samples. Pooled over reps [1, 2, 3]: dagger_b30 3x128 4.28 vs d_termq 3x256 4.76; dagger_b30 > d_termq in 0/6 cells on the mean (see §10). |
| offline_v1_01 planner 5.63; offline_v1_02 (identical rerun) 4.81; offline_v1 4.69 (EXPERIMENTS §4) | final, rep 1 | thr 0.1 pooled mpc final 5.08 ± 0.34 (n=6), tail 4.67 ± 0.23 (n=6); only fired runs tail 4.68 ± 0.26 (n=5); offline_v1 (0.15) tail 4.66 ± 0.15 (n=3); rung-D config (no offline) tail 4.61 ± 0.43 (n=6) | 5.63 is the top draw. The offline phase gives no clear planner or policy gain over the rung-D config; `offline_v1_02_rep3` never triggered the phase (extra updates 0). |
| offline_v2 periodic planner 3.94 and 4.44 for r1/r2 (EXPERIMENTS §5) | final | periodic 4.25 ± 0.27 (n=3); baseline 2.42 ± 2.14 (n=3) (excl. degenerate r1 3.62 ± 0.62 (n=2)); extra updates periodic r1/r2/r3 = 8000/0/0 | Numbers reproduce, but the periodic phase **never fired in r2 and r3** and fired once (8000 updates) in r1, so 'periodic' is mostly the baseline. No condition is separable from the others with n=3. |
| pure_rl_v1: all three reps 0 goals/ep, drop 1.0 (EXPERIMENTS §6) | all reps | pi final 0.00 ± 0.00 (n=3), drop 1.00 ± 0.00 (n=3) | Holds (n=3). |
| SysID fixes the contact model: comfree err 0.19→0.09 (C), 0.19→0.11 (D); joint 0.085→0.079; inertial worse 0.092→0.107 (C) (README) | rep 1 | C reps comfree 0.188→0.092, 0.176→0.075, 0.185→0.076; joint 0.085→0.079, 0.085→0.086, 0.084→0.082; inertial 0.092→0.107, 0.079→0.123, 0.085→0.124; D comfree 0.186→0.112, 0.186→0.067, 0.179→0.069 | **Holds (n=3)** for C and D: the comfree group roughly halves in every rep, joint stays flat, inertial gets worse in every C rep. Same pattern in every 150k/300k config (§12). Exception: in the 60k offline_v2 baseline/forced_early runs the comfree error has not moved yet. |

## 14. Hero checkpoint

**Criterion.** Eligible: `agent.pt` and `videos/` on disk, not degenerate, planner drop rate ≤ 0.1 both in the final eval and in the tail. Rank by tail mpc goals/ep; every run within 0.25 goals/ep of the best counts as tied (well inside one 16-episode eval's noise), and among those the highest pi tail goals/ep wins, so the policy also works. 50 of 78 runs are eligible (all runs have `agent.pt` and videos on disk).

| rank | run | env steps | mpc G tail | mpc drop tail | pi G tail | mpc G final | pi G final |
|---|---|---|---|---|---|---|---|
| 1 | `ablation_300k_rep2/e_bc` (tie band) | 300000 | 5.54 | 0.00 | 1.38 | 5.81 | 1.88 |
| 2 | `ablation_300k_rep3/e_bc` (tie band) | 300000 | 5.71 | 0.00 | 1.08 | 6.12 | 1.12 |
| 3 | `ablation_300k/e_bc` (tie band) | 300000 | 5.54 | 0.00 | 0.89 | 5.50 | 0.50 |
| 4 | `ablation_300k/d_termq` | 300000 | 5.40 | 0.02 | 1.14 | 5.62 | 1.12 |
| 5 | `ablation_rep2/e_bc` | 150000 | 5.33 | 0.02 | 0.83 | 5.25 | 0.56 |
| 6 | `ablation_300k_rep2/d_termq` | 300000 | 5.29 | 0.00 | 2.08 | 5.25 | 2.38 |
| 7 | `vec/netsize_default` | 150000 | 5.21 | 0.00 | 0.54 | 5.75 | 0.75 |
| 8 | `ablation_rep3/d_termq` | 150000 | 4.98 | 0.02 | 1.08 | 5.12 | 1.19 |

**Hero: `ablation_300k_rep2/e_bc`**. Runner-up: `ablation_300k_rep3/e_bc`. Best at the 150k main budget: `ablation_rep2/e_bc`. Note: `ablation_300k_rep2/d_termq` has the best policy among runs with mpc tail ≥ 5 (pi tail 2.08, mpc tail 5.29) but falls just outside the tie band; it is the pick if the policy video matters more than the planner video.

Hero videos (GIF, recorded at the eval of that env step; checkpoint `comfree_tdmpc/outputs/ablation_300k_rep2/e_bc/agent.pt`, 14.3 MB = final weights):

| file | step | mode | size MB |
|---|---|---|---|
| `comfree_tdmpc/outputs/ablation_300k_rep2/e_bc/videos/s0050080_mpc.gif` | 50080 | mpc | 3.70 |
| `comfree_tdmpc/outputs/ablation_300k_rep2/e_bc/videos/s0050080_pi.gif` | 50080 | pi | 2.80 |
| `comfree_tdmpc/outputs/ablation_300k_rep2/e_bc/videos/s0100160_mpc.gif` | 100160 | mpc | 3.67 |
| `comfree_tdmpc/outputs/ablation_300k_rep2/e_bc/videos/s0100160_pi.gif` | 100160 | pi | 3.20 |
| `comfree_tdmpc/outputs/ablation_300k_rep2/e_bc/videos/s0150240_mpc.gif` | 150240 | mpc | 3.82 |
| `comfree_tdmpc/outputs/ablation_300k_rep2/e_bc/videos/s0150240_pi.gif` | 150240 | pi | 4.27 |
| `comfree_tdmpc/outputs/ablation_300k_rep2/e_bc/videos/s0200320_mpc.gif` | 200320 | mpc | 3.89 |
| `comfree_tdmpc/outputs/ablation_300k_rep2/e_bc/videos/s0200320_pi.gif` | 200320 | pi | 3.96 |
| `comfree_tdmpc/outputs/ablation_300k_rep2/e_bc/videos/s0250400_mpc.gif` | 250400 | mpc | 3.56 |
| `comfree_tdmpc/outputs/ablation_300k_rep2/e_bc/videos/s0250400_pi.gif` | 250400 | pi | 4.08 |
| `comfree_tdmpc/outputs/ablation_300k_rep2/e_bc/videos/s0300000_mpc.gif` | 300000 | mpc | 3.51 |
| `comfree_tdmpc/outputs/ablation_300k_rep2/e_bc/videos/s0300000_pi.gif` | 300000 | pi | 3.71 |

Runner-up videos (GIF, recorded at the eval of that env step; checkpoint `comfree_tdmpc/outputs/ablation_300k_rep3/e_bc/agent.pt`, 14.3 MB = final weights):

| file | step | mode | size MB |
|---|---|---|---|
| `comfree_tdmpc/outputs/ablation_300k_rep3/e_bc/videos/s0050080_mpc.gif` | 50080 | mpc | 3.60 |
| `comfree_tdmpc/outputs/ablation_300k_rep3/e_bc/videos/s0050080_pi.gif` | 50080 | pi | 3.35 |
| `comfree_tdmpc/outputs/ablation_300k_rep3/e_bc/videos/s0100160_mpc.gif` | 100160 | mpc | 3.56 |
| `comfree_tdmpc/outputs/ablation_300k_rep3/e_bc/videos/s0100160_pi.gif` | 100160 | pi | 2.88 |
| `comfree_tdmpc/outputs/ablation_300k_rep3/e_bc/videos/s0150240_mpc.gif` | 150240 | mpc | 3.69 |
| `comfree_tdmpc/outputs/ablation_300k_rep3/e_bc/videos/s0150240_pi.gif` | 150240 | pi | 3.56 |
| `comfree_tdmpc/outputs/ablation_300k_rep3/e_bc/videos/s0200320_mpc.gif` | 200320 | mpc | 3.48 |
| `comfree_tdmpc/outputs/ablation_300k_rep3/e_bc/videos/s0200320_pi.gif` | 200320 | pi | 3.56 |
| `comfree_tdmpc/outputs/ablation_300k_rep3/e_bc/videos/s0250400_mpc.gif` | 250400 | mpc | 3.88 |
| `comfree_tdmpc/outputs/ablation_300k_rep3/e_bc/videos/s0250400_pi.gif` | 250400 | pi | 3.55 |
| `comfree_tdmpc/outputs/ablation_300k_rep3/e_bc/videos/s0300000_mpc.gif` | 300000 | mpc | 3.67 |
| `comfree_tdmpc/outputs/ablation_300k_rep3/e_bc/videos/s0300000_pi.gif` | 300000 | pi | 4.07 |

## Appendix A. Every run

### ablation

| run | env steps | mpc G final | mpc G tail | mpc drop final | mpc drop tail | mpc R tail | pi G final | pi G tail | pi drop final | pi drop tail | pi R tail | wall h | extra updates | drift ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ablation/a_base` | 150000 | 1.69 | 1.62 | 0.81 | 0.75 | 10 | 0.69 | 0.57 | 0.94 | 0.89 | -161 | 1.02 | 0 | 1.05 |
| `ablation/b_distill` | 150000 | 2.19 | 2.02 | 0.69 | 0.69 | 89 | 1.31 | 1.14 | 0.81 | 0.73 | -65 | 1.02 | 0 | 1.07 |
| `ablation/c_sysid` | 150000 | 1.94 | 1.94 | 0.81 | 0.88 | 199 | 1.44 | 1.25 | 0.69 | 0.67 | -2 | 1.09 | 0 | 1.25 |
| `ablation/d_termq` | 150000 | 4.75 | 4.56 | 0.06 | 0.02 | 942 | 0.56 | 0.65 | 0.50 | 0.51 | -124 | 1.13 | 0 | 2.04 |
| `ablation/e_bc` | 150000 | 5.06 | 4.69 | 0.06 | 0.06 | 988 | 1.31 | 0.65 | 0.12 | 0.20 | -60 | 1.13 | 0 | 2.35 |
| `ablation/f_meantarget` | 150000 | 5.19 | 4.40 | 0.00 | 0.04 | 918 | 1.06 | 0.93 | 0.12 | 0.20 | 7 | 1.13 | 0 | 2.17 |
| `ablation_rep2/a_base` | 150000 | 1.44 | 1.58 | 0.81 | 0.75 | 48 | 0.62 | 0.62 | 0.75 | 0.70 | -170 | 0.80 | 0 | NA |
| `ablation_rep2/b_distill` | 150000 | 1.62 | 1.56 | 0.94 | 0.90 | 45 | 0.62 | 0.86 | 0.94 | 0.81 | -93 | 0.79 | 0 | NA |
| `ablation_rep2/c_sysid` | 150000 | 3.12 | 2.77 | 0.75 | 0.81 | 402 | 1.62 | 1.23 | 0.81 | 0.79 | -1 | 0.85 | 0 | NA |
| `ablation_rep2/d_termq` | 150000 | 4.38 | 4.62 | 0.00 | 0.02 | 930 | 0.69 | 0.57 | 0.44 | 0.47 | -118 | 0.88 | 0 | NA |
| `ablation_rep2/e_bc` | 150000 | 5.25 | 5.33 | 0.06 | 0.02 | 1071 | 0.56 | 0.83 | 0.19 | 0.28 | -19 | 0.87 | 0 | NA |
| `ablation_rep2/f_meantarget` | 150000 | 5.56 | 4.90 | 0.00 | 0.02 | 971 | 1.19 | 1.08 | 0.06 | 0.19 | 47 | 0.88 | 0 | NA |
| `ablation_rep3/a_base` | 150000 | 1.31 | 1.10 | 0.88 | 0.69 | -114 | 0.00 | 0.00 | 1.00 | 1.00 | -2462 | 0.97 | 0 | NA |
| `ablation_rep3/b_distill` | 150000 | 1.69 | 1.81 | 0.75 | 0.77 | 88 | 1.12 | 0.83 | 0.94 | 0.85 | -121 | 0.99 | 0 | NA |
| `ablation_rep3/c_sysid` | 150000 | 3.19 | 2.98 | 0.62 | 0.62 | 498 | 0.50 | 0.76 | 0.81 | 0.64 | -87 | 1.08 | 0 | NA |
| `ablation_rep3/d_termq` | 150000 | 5.12 | 4.98 | 0.00 | 0.02 | 1051 | 1.19 | 1.08 | 0.44 | 0.44 | 11 | 1.10 | 0 | NA |
| `ablation_rep3/e_bc` | 150000 | 5.00 | 4.71 | 0.06 | 0.04 | 955 | 1.00 | 0.93 | 0.12 | 0.29 | 30 | 1.10 | 0 | NA |
| `ablation_rep3/f_meantarget` | 150000 | 4.56 | 4.42 | 0.00 | 0.04 | 891 | 1.06 | 0.97 | 0.25 | 0.23 | 30 | 1.09 | 0 | NA |

### ablation_300k

| run | env steps | mpc G final | mpc G tail | mpc drop final | mpc drop tail | mpc R tail | pi G final | pi G tail | pi drop final | pi drop tail | pi R tail | wall h | extra updates | drift ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ablation_300k/d_termq` | 300000 | 5.62 | 5.40 | 0.00 | 0.02 | 1216 | 1.12 | 1.14 | 0.50 | 0.47 | 51 | 6.66 | 0 | NA |
| `ablation_300k/e_bc` | 300000 | 5.50 | 5.54 | 0.00 | 0.00 | 1225 | 0.50 | 0.89 | 0.19 | 0.17 | 63 | 4.30 | 0 | NA |
| `ablation_300k_rep2/d_termq` | 300000 | 5.25 | 5.29 | 0.00 | 0.00 | 1140 | 2.38 | 2.08 | 0.31 | 0.39 | 269 | 1.81 | 0 | NA |
| `ablation_300k_rep2/e_bc` | 300000 | 5.81 | 5.54 | 0.00 | 0.00 | 1188 | 1.88 | 1.38 | 0.25 | 0.25 | 180 | 2.23 | 0 | NA |
| `ablation_300k_rep3/d_termq` ⚠degenerate | 300000 | 0.00 | 0.00 | 1.00 | 1.00 | -2422 | 0.00 | 0.00 | 1.00 | 1.00 | -2441 | 2.22 | 0 | NA |
| `ablation_300k_rep3/e_bc` | 300000 | 6.12 | 5.71 | 0.00 | 0.00 | 1256 | 1.12 | 1.08 | 0.38 | 0.20 | 105 | 2.23 | 0 | NA |

### dagger

| run | env steps | mpc G final | mpc G tail | mpc drop final | mpc drop tail | mpc R tail | pi G final | pi G tail | pi drop final | pi drop tail | pi R tail | wall h | extra updates | drift ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vec/d_termq_s128` | 150000 | 4.44 | 4.00 | 0.00 | 0.00 | 720 | 1.25 | 0.76 | 0.44 | 0.46 | -102 | 0.95 | 0 | 1.79 |
| `vec/d_termq_s128_rep2` | 150000 | 4.50 | 4.23 | 0.00 | 0.02 | 795 | 0.88 | 0.83 | 0.12 | 0.34 | -75 | 0.84 | 0 | NA |
| `vec/d_termq_s128_rep3` | 150000 | 4.75 | 4.27 | 0.00 | 0.04 | 844 | 0.50 | 0.73 | 0.44 | 0.35 | -30 | 0.83 | 0 | NA |
| `vec/dagger_b0` | 150000 | 4.44 | 4.27 | 0.00 | 0.02 | 886 | 0.31 | 0.49 | 0.19 | 0.25 | -72 | 2.72 | 0 | 2.73 |
| `vec/dagger_b0_rep2` | 150000 | 4.62 | 4.54 | 0.00 | 0.00 | 941 | 0.75 | 0.57 | 0.44 | 0.26 | -69 | 1.09 | 0 | NA |
| `vec/dagger_b0_rep3` | 150000 | 4.56 | 4.77 | 0.06 | 0.06 | 968 | 0.69 | 0.80 | 0.12 | 0.26 | -1 | 1.09 | 0 | NA |
| `vec/dagger_b30` | 150000 | 4.69 | 4.81 | 0.06 | 0.04 | 961 | 1.38 | 0.79 | 0.25 | 0.26 | 2 | 2.14 | 0 | 2.01 |
| `vec/dagger_b30_rep2` | 150000 | 4.31 | 4.04 | 0.06 | 0.08 | 788 | 0.50 | 0.62 | 0.06 | 0.16 | -22 | 1.10 | 0 | NA |
| `vec/dagger_b30_rep3` | 150000 | 4.25 | 4.02 | 0.00 | 0.00 | 821 | 0.69 | 0.67 | 0.06 | 0.14 | 6 | 1.10 | 0 | NA |
| `vec/dagger_b30_s128` | 150000 | 3.25 | 3.44 | 0.12 | 0.04 | 572 | 0.69 | 0.67 | 0.12 | 0.16 | -103 | 0.92 | 0 | 1.56 |
| `vec/dagger_b30_s128_rep2` | 150000 | 4.12 | 3.85 | 0.00 | 0.00 | 712 | 1.12 | 0.90 | 0.19 | 0.19 | 55 | 0.83 | 0 | NA |
| `vec/dagger_b30_s128_rep3` | 150000 | 3.88 | 3.77 | 0.06 | 0.10 | 646 | 1.44 | 1.06 | 0.19 | 0.19 | 37 | 0.83 | 0 | NA |
| `vec/dagger_b60` | 150000 | 3.31 | 3.75 | 0.12 | 0.10 | 701 | 0.44 | 1.12 | 0.38 | 0.19 | 37 | 2.08 | 0 | 1.79 |
| `vec/dagger_b60_rep2` | 150000 | 4.06 | 3.92 | 0.00 | 0.15 | 734 | 0.75 | 0.72 | 0.31 | 0.15 | -9 | 1.09 | 0 | NA |
| `vec/dagger_b60_rep3` | 150000 | 4.06 | 3.71 | 0.06 | 0.02 | 656 | 1.19 | 0.83 | 0.31 | 0.20 | -35 | 1.09 | 0 | NA |

### no_q

| run | env steps | mpc G final | mpc G tail | mpc drop final | mpc drop tail | mpc R tail | pi G final | pi G tail | pi drop final | pi drop tail | pi R tail | wall h | extra updates | drift ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vec/e_no_q` | 150000 | 5.38 | 4.96 | 0.00 | 0.00 | 977 | 0.25 | 0.54 | 0.19 | 0.26 | -54 | 1.00 | 0 | 2.57 |
| `vec/e_no_q_rep2` | 150000 | 4.88 | 4.56 | 0.00 | 0.04 | 903 | 0.81 | 0.50 | 0.25 | 0.18 | -87 | 1.08 | 0 | NA |
| `vec/e_no_q_rep3` ⚠degenerate | 150000 | 0.00 | 0.00 | 1.00 | 1.00 | -2473 | 0.00 | 0.00 | 1.00 | 1.00 | -2464 | 1.07 | 0 | NA |
| `vec/f_no_q` | 150000 | 4.81 | 4.83 | 0.00 | 0.00 | 1041 | 0.88 | 0.68 | 0.19 | 0.36 | -53 | 2.07 | 0 | 2.25 |
| `vec/f_no_q_rep2` | 150000 | 4.25 | 4.27 | 0.00 | 0.00 | 917 | 0.06 | 0.84 | 0.25 | 0.22 | -6 | 1.07 | 0 | NA |
| `vec/f_no_q_rep3` | 150000 | 4.69 | 4.23 | 0.12 | 0.06 | 835 | 0.75 | 0.59 | 0.25 | 0.22 | -32 | 1.07 | 0 | NA |

### netsize

| run | env steps | mpc G final | mpc G tail | mpc drop final | mpc drop tail | mpc R tail | pi G final | pi G tail | pi drop final | pi drop tail | pi R tail | wall h | extra updates | drift ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vec/netsize_default` | 150000 | 5.75 | 5.21 | 0.00 | 0.00 | 1067 | 0.75 | 0.54 | 0.19 | 0.36 | -88 | 3.75 | 0 | 2.23 |
| `vec/netsize_default_rep2` | 150000 | 4.44 | 4.17 | 0.06 | 0.02 | 874 | 0.56 | 0.46 | 0.69 | 0.56 | -152 | 1.09 | 0 | NA |
| `vec/netsize_default_rep3` | 150000 | 3.88 | 4.12 | 0.00 | 0.06 | 837 | 0.69 | 0.62 | 0.50 | 0.49 | -63 | 1.10 | 0 | NA |
| `vec/netsize_large` | 150000 | 4.38 | 4.25 | 0.06 | 0.04 | 875 | 0.75 | 0.56 | 0.69 | 0.48 | -117 | 2.22 | 0 | 1.95 |
| `vec/netsize_large_rep2` | 150000 | 3.81 | 4.08 | 0.06 | 0.04 | 829 | 1.12 | 0.97 | 0.38 | 0.44 | -25 | 1.12 | 0 | NA |
| `vec/netsize_large_rep3` | 150000 | 4.81 | 4.56 | 0.00 | 0.02 | 910 | 0.94 | 0.76 | 0.44 | 0.40 | -8 | 1.11 | 0 | NA |
| `vec/netsize_small` | 150000 | 4.81 | 4.69 | 0.12 | 0.06 | 917 | 0.19 | 0.47 | 0.38 | 0.23 | -59 | 3.68 | 0 | 2.46 |
| `vec/netsize_small_rep2` | 150000 | 4.31 | 4.44 | 0.00 | 0.02 | 921 | 0.62 | 0.47 | 0.19 | 0.40 | -110 | 1.10 | 0 | NA |
| `vec/netsize_small_rep3` | 150000 | 4.75 | 4.56 | 0.00 | 0.06 | 929 | 0.38 | 0.24 | 0.31 | 0.31 | -151 | 1.11 | 0 | NA |

### offline_v1

| run | env steps | mpc G final | mpc G tail | mpc drop final | mpc drop tail | mpc R tail | pi G final | pi G tail | pi drop final | pi drop tail | pi R tail | wall h | extra updates | drift ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vec/offline_v1` | 150000 | 4.69 | 4.50 | 0.00 | 0.02 | 918 | 0.69 | 0.54 | 0.50 | 0.45 | -94 | 4.63 | 50000 | 2.12 |
| `vec/offline_v1_01` | 150000 | 5.62 | 4.75 | 0.00 | 0.04 | 1011 | 1.31 | 0.98 | 0.44 | 0.47 | 15 | 4.08 | 50000 | 2.18 |
| `vec/offline_v1_01_rep2` | 150000 | 5.31 | 4.67 | 0.00 | 0.04 | 971 | 0.81 | 0.71 | 0.62 | 0.50 | -94 | 5.04 | 50000 | NA |
| `vec/offline_v1_01_rep3` | 150000 | 4.75 | 4.27 | 0.00 | 0.02 | 884 | 0.75 | 0.97 | 0.62 | 0.53 | -24 | 5.07 | 50000 | NA |
| `vec/offline_v1_02` | 150000 | 4.81 | 4.98 | 0.00 | 0.00 | 1036 | 0.88 | 0.88 | 0.50 | 0.50 | -10 | 3.96 | 50000 | 2.27 |
| `vec/offline_v1_02_rep2` | 150000 | 4.88 | 4.75 | 0.06 | 0.04 | 933 | 0.38 | 0.66 | 0.50 | 0.53 | -81 | 5.15 | 50000 | NA |
| `vec/offline_v1_02_rep3` | 150000 | 5.12 | 4.60 | 0.00 | 0.00 | 959 | 0.69 | 0.49 | 0.44 | 0.51 | -147 | 1.11 | 0 | NA |
| `vec/offline_v1_rep2` | 150000 | 5.06 | 4.79 | 0.00 | 0.00 | 970 | 0.25 | 0.52 | 0.56 | 0.59 | -115 | 5.04 | 50000 | NA |
| `vec/offline_v1_rep3` | 150000 | 5.06 | 4.69 | 0.00 | 0.02 | 1006 | 1.25 | 0.93 | 0.56 | 0.53 | -18 | 4.99 | 50000 | NA |

### offline_v2

| run | env steps | mpc G final | mpc G tail | mpc drop final | mpc drop tail | mpc R tail | pi G final | pi G tail | pi drop final | pi drop tail | pi R tail | wall h | extra updates | drift ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vec/offline_v2_baseline_r1` ⚠degenerate | 60000 | 0.00 | 0.00 | 1.00 | 1.00 | -2519 | 0.00 | 0.00 | 1.00 | 1.00 | -2543 | 0.42 | 0 | NA |
| `vec/offline_v2_baseline_r2` | 60000 | 3.19 | 3.04 | 0.06 | 0.25 | 466 | 0.69 | 0.62 | 0.69 | 0.49 | -97 | 0.47 | 0 | 1.54 |
| `vec/offline_v2_baseline_r3` | 60000 | 4.06 | 3.77 | 0.06 | 0.08 | 653 | 0.31 | 0.35 | 0.69 | 0.61 | -256 | 0.43 | 0 | NA |
| `vec/offline_v2_forced_early_r1` | 60000 | 3.94 | 3.67 | 0.12 | 0.12 | 667 | 0.31 | 0.20 | 0.50 | 0.53 | -274 | 4.23 | 50000 | 1.75 |
| `vec/offline_v2_forced_early_r2` | 60000 | 3.38 | 3.00 | 0.00 | 0.25 | 475 | 0.44 | 0.39 | 0.56 | 0.55 | -173 | 4.26 | 50000 | 1.81 |
| `vec/offline_v2_forced_early_r3` | 60000 | 3.62 | 3.52 | 0.12 | 0.12 | 625 | 0.25 | 0.21 | 0.62 | 0.66 | -298 | 3.39 | 50000 | NA |
| `vec/offline_v2_forced_late_r1` | 60000 | 3.44 | 3.50 | 0.19 | 0.08 | 664 | 0.56 | 0.60 | 0.56 | 0.45 | -114 | 4.47 | 50000 | 1.71 |
| `vec/offline_v2_forced_late_r2` | 60000 | 3.62 | 3.67 | 0.00 | 0.04 | 733 | 0.25 | 0.24 | 0.38 | 0.48 | -158 | 4.44 | 50000 | 1.57 |
| `vec/offline_v2_forced_late_r3` | 60000 | 4.12 | 3.56 | 0.00 | 0.21 | 661 | 0.19 | 0.27 | 0.50 | 0.53 | -204 | 3.47 | 50000 | NA |
| `vec/offline_v2_periodic_r1` | 60000 | 3.94 | 3.44 | 0.06 | 0.10 | 581 | 0.50 | 0.49 | 0.62 | 0.46 | -142 | 1.11 | 8000 | 1.55 |
| `vec/offline_v2_periodic_r2` | 60000 | 4.44 | 3.62 | 0.00 | 0.02 | 704 | 0.31 | 0.21 | 0.44 | 0.40 | -208 | 0.47 | 0 | 1.73 |
| `vec/offline_v2_periodic_r3` | 60000 | 4.38 | 3.79 | 0.00 | 0.04 | 714 | 0.50 | 0.79 | 0.38 | 0.38 | -84 | 0.34 | 0 | NA |

### pure_rl

| run | env steps | mpc G final | mpc G tail | mpc drop final | mpc drop tail | mpc R tail | pi G final | pi G tail | pi drop final | pi drop tail | pi R tail | wall h | extra updates | drift ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vec/pure_rl_v1` | 150000 | NA | NA | NA | NA | NA | 0.00 | 0.00 | 1.00 | 1.00 | -2483 | 0.24 | 0 | NA |
| `vec/pure_rl_v1_rep2` | 150000 | NA | NA | NA | NA | NA | 0.00 | 0.00 | 1.00 | 1.00 | -2557 | 0.29 | 0 | NA |
| `vec/pure_rl_v1_rep3` | 150000 | NA | NA | NA | NA | NA | 0.00 | 0.00 | 1.00 | 1.00 | -2440 | 0.29 | 0 | NA |
