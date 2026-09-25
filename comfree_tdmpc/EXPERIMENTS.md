# Experiments

Reproduction index for everything in `outputs/`. Each training command below
was reconstructed from the run's `config.json` and cross-checked against the
driver scripts `rerun_ablation_variance.sh` and `run_offline_v2_sweep.sh`,
which are the exact record for the runs they launched.

Conventions:

- Run all commands from `comfree_tdmpc/` in the environment described in the
  root README, with `MUJOCO_GL=egl` set (training runs render evaluation
  videos).
- Every training run uses `--seed 0` (the default) and the same reality
  (`--reality_seed 12345 --reality_perturb 0.25`, defaults). The GPU simulator
  is not bitwise deterministic, so repetitions (`_rep2`, `_rep3`, `_r1`..`_r3`)
  are same-seed reruns and differ from each other; they serve as the variance
  estimate.
- All `train_vec.py` defaults not listed in a command were in effect
  (`--num_envs 16 --horizon 8 --iterations 3 --num_samples 256 --num_elites 32
  --num_pi_trajs 24`, 5k-step policy evaluations, 25k-step planner
  evaluations). `--tag` only sets the label burned into videos.
- Result figures quoted here are single final evaluations (16 episodes) of the
  first repetition unless stated otherwise, and are noisy. The ablation-ladder
  table with tail means is in [README.md](README.md).
- Each run directory holds `config.json`, `train.csv`, `eval.csv`,
  `sysid.csv`, `params.npy`, `params_reality.npy` and, where the diagnostic was
  run, `diagnosis.json`. Checkpoints, videos and console logs are not in the
  repository.

## Hardware and runtime

The ablation ladder in `outputs/ablation/` was produced on an RTX 4080; the
300k-step runs, `dagger_b0/b30/b60` and `f_no_q` on an RTX 3070; the
repetitions and the remaining runs on an RTX 4090. A 150k-step run with the
planner takes roughly 50-70 min on the RTX 4080/4090 (about 37-47 env
steps/s); a 300k-step run took about 6.7 h on the RTX 3070; runs in which the
offline phase fires take several hours longer; `--pure_rl` takes about 20 min.
Jobs are meant to run one at a time: 16 environments plus the planner fill one
GPU.

## 1. Milestone and benchmark scripts

| script | question | command | output |
| --- | --- | --- | --- |
| `00_smoke_test.py` (M0) | does the stack run; throughput | `python scripts/00_smoke_test.py --nworlds 1 4` | console (`--render` writes `outputs/00_smoke/`) |
| `01_planner_only.py` (M1) | can MPPI through ComFree solve the task without learning | `python scripts/01_planner_only.py` | `outputs/01_planner/` |
| `02_sysid_recovery.py` (M2) | does 4-parameter central-difference SysID recover the truth | `python scripts/02_sysid_recovery.py` | `outputs/02_sysid/` |
| `03_tune_mppi.py` | single-env MPPI hyperparameters, quality per second | `python scripts/03_tune_mppi.py` | `outputs/03_tune_mppi/` |
| `04_check_smoothness.py` | is parameter -> trajectory smooth enough for first-order SysID | `python scripts/04_check_smoothness.py` | `outputs/report/fig5_smoothness.png`, `smoothness.npz` |
| `05_sysid_highdim.py` | does stochastic SysID identify the 71-parameter set | `python scripts/05_sysid_highdim.py` | `outputs/05_sysid_highdim/` |
| `06_throughput.py` | where the wall clock goes (sim, plan, update) | `python scripts/06_throughput.py` | `outputs/06_throughput/throughput.json` |
| `07_verify_vec.py` | does the vectorised stack match the single-env reference; speedup | `python scripts/07_verify_vec.py` | `outputs/07_verify/vec.json` |
| `08_tune_planner_vec.py` | cheapest vectorised planner configuration per second | `python scripts/08_tune_planner_vec.py` | `outputs/08_tune/planner.json` |

Only the outputs of 06, 07 and 08 are in the repository; their contents match
the scripts' default arguments. Recorded results:
`throughput.json` reports 92.7k control steps/s (0.93M physics steps/s) at 512
worlds; `vec.json` passes all correctness checks (vectorised env vs single env
3.0e-8 after one step) and reports a 4.2x environment-step speedup at 16
environments over one; `planner.json` holds 9 planner configurations.

## 2. Ablation ladder, 150k steps

```bash
python scripts/run_ablation.py --steps 150000
```

Runs the rungs sequentially into `outputs/ablation/<rung>/` (console output in
`outputs/ablation/<rung>.log`); finished rungs are skipped. Each rung calls
`train_vec.py --name <rung> --out_root outputs/ablation --tag <rung> --steps 150000`
plus the flags below (cumulative).

| rung | added flags (`train_vec.py`) |
| --- | --- |
| `a_base` | none |
| `b_distill` | `--distill` |
| `c_sysid` | `--distill --sysid` |
| `d_termq` | `--distill --sysid --terminal_value` |
| `e_bc` | `--distill --sysid --terminal_value --actor_mode bc --sim_data_ratio 0.75` |
| `f_meantarget` | `--distill --sysid --terminal_value --actor_mode bc_mu --elite_target mean --sim_data_ratio 0.75 --prior_coef 2.0` |

Repetitions (launched by `rerun_ablation_variance.sh`):

```bash
python scripts/run_ablation.py --steps 150000 --out_root outputs/ablation_rep2 --only a_base b_distill c_sysid d_termq e_bc
python scripts/run_ablation.py --steps 150000 --out_root outputs/ablation_rep2 --only f_meantarget
# same with outputs/ablation_rep3
```

Results: see the table in [README.md](README.md) (`outputs/report/figures/summary.json`).

## 3. Ablation rungs D and E at 300k steps

Were the policy curves still climbing at 150k?

```bash
python scripts/run_ablation.py --steps 300000 --out_root outputs/ablation_300k --only d_termq e_bc
python scripts/run_ablation.py --steps 300000 --out_root outputs/ablation_300k_rep2 --only d_termq e_bc
python scripts/run_ablation.py --steps 300000 --out_root outputs/ablation_300k_rep3 --only d_termq e_bc
```

The first command is reconstructed from the configs (tags, output directories
and flags match); the repetitions are from `rerun_ablation_variance.sh`.
Result (rep 1, final evaluation): `d_termq` planner 5.62 goals/ep (4.56 tail
mean at 150k); `e_bc` policy 0.50 goals/ep.

## 4. Vectorised experiments (`outputs/vec/`)

Every row is `python scripts/train_vec.py --name <name> --steps 150000 <flags>`,
output in `outputs/vec/<name>/` (`--out_root` defaults to `outputs/vec`). Repetitions `<name>_rep2` and `<name>_rep3`
were launched by `rerun_ablation_variance.sh` as
`python scripts/train_vec.py --name <name>_repN --out_root outputs/vec --tag <name>_repN <same flags>`.

| name | flags | question | result (rep 1, final eval) |
| --- | --- | --- | --- |
| `netsize_small` | `--distill --sysid --terminal_value --mlp_dim 128 --latent_dim 128` | does policy capacity limit distillation? | policy 0.19 goals/ep |
| `netsize_default` | `--distill --sysid --terminal_value` | (same configuration as rung D) | policy 0.75 |
| `netsize_large` | `--distill --sysid --terminal_value --mlp_dim 512 --latent_dim 512` | | policy 0.75 |
| `e_no_q` | `--distill --sysid --terminal_value --actor_mode bc_no_q --sim_data_ratio 0.75` | rung E without the Q term in the actor loss | |
| `f_no_q` | `--distill --sysid --terminal_value --actor_mode bc_mu_no_q --elite_target mean --sim_data_ratio 0.75 --prior_coef 2.0` | rung F without the Q term in the actor loss | no improvement over F; drift ratio 2.25 vs 2.17 (`diagnosis.json`) |
| `offline_v1` | `--distill --sysid --terminal_value --offline_start_val_loss 0.15` | one-shot offline consolidation phase (`docs/offline_training_design.md`) | planner 4.69 |
| `offline_v1_01` | `--distill --sysid --terminal_value --offline_start_val_loss 0.1` | same, lower trigger threshold | planner 5.63 |
| `offline_v1_02` | `--distill --sysid --terminal_value --offline_start_val_loss 0.1` | identical rerun of `offline_v1_01` | planner 4.81 |
| `dagger_b0` | `--distill --sysid --terminal_value --actor_mode bc_mu --dagger_beta 0.0` | DAgger-style relabelling: probability of executing the policy's action | policy 0.31 |
| `dagger_b30` | `--distill --sysid --terminal_value --actor_mode bc_mu --dagger_beta 0.3` | | policy 1.38 |
| `dagger_b60` | `--distill --sysid --terminal_value --actor_mode bc_mu --dagger_beta 0.6` | | policy 0.44 |
| `dagger_b30_s128` | `--distill --sysid --terminal_value --actor_mode bc_mu --dagger_beta 0.3 --num_samples 128` | training with half the planner samples | policy 0.69, planner drop rate 0.125 |
| `d_termq_s128` | `--distill --sysid --terminal_value --num_samples 128` | | policy 1.25, planner drop rate 0.0 |

## 5. Offline phase v2: trigger sweep, 60k steps

When should the offline phase fire? Four conditions, three identically
configured samples each, all with `--distill --sysid --terminal_value --steps 60000`:

| condition | extra flags |
| --- | --- |
| `baseline` | none (offline phase off) |
| `forced_early` | `--offline_force_step 20000 --offline_mode once` |
| `forced_late` | `--offline_force_step 45000 --offline_mode once` |
| `periodic` | `--offline_start_val_loss 0.15 --offline_confirm_checks 2 --offline_mode periodic --offline_periodic_num_updates 8000` |

```bash
python scripts/train_vec.py --name offline_v2_<condition>_r<k> --out_root outputs/vec \
    --distill --sysid --terminal_value --steps 60000 <extra flags>
```

`run_offline_v2_sweep.sh` launched `forced_early`, `forced_late` and
`periodic` r1-r2 and `baseline` r2; `rerun_ablation_variance.sh` (section 9 of
the script) added `baseline` r1 and all r3 runs. Output:
`outputs/vec/offline_v2_<condition>_r<k>/`. Result (final eval): `periodic`
planner 3.94 and 4.44 goals/ep for r1 and r2.

## 6. Pure RL (no planner)

```bash
python scripts/train_vec.py --name pure_rl_v1 --pure_rl --steps 150000
```

Plain SAC-style actor-critic acting directly in the environment (`--pure_rl`
forces `--actor_mode sac` and is exclusive with the planner switches).
Repetitions `pure_rl_v1_rep2/_rep3` via `rerun_ablation_variance.sh`. Result:
all three repetitions reach 0 goals/ep with drop rate 1.0.

## 7. Rollout-budget ablation (evaluation only)

```bash
python scripts/report/ablate_rollout_budget.py
```

Loads the checkpoints `outputs/vec/dagger_b30` and `outputs/ablation/d_termq`
(`--runs` to change) and sweeps `(iterations, num_samples)` at fixed
`num_pi_trajs=24`, 32 episodes each, seed 77. Output:
`outputs/rollout_budget/results.json` and `rollout_budget.png`. Needs the
`agent.pt` checkpoints, which are not in the repository. Result:
`dagger_b30` at 3 x 128 samples reaches 4.81 goals/ep at 146 ms per plan call,
versus 4.69 goals/ep at 221 ms for `d_termq` at the full 3 x 256.

## 8. Variance reruns

```bash
PY=python ./rerun_ablation_variance.sh
```

Reruns sections 2, 3, 4 and 6 twice more (rep2, rep3) and tops up section 5 to
three samples per condition, one job at a time. It is resumable (a job whose
`config.json` exists is skipped) and writes its progress to
`outputs/variance_reruns/driver.log`. `run_offline_v2_sweep.sh` is the
original section-5 driver; both scripts use `PY="${PY:-python}"`.

Three runs degenerated from the first evaluation on (every evaluation episode,
planner included, drops the cube): `ablation_300k_rep3/d_termq`,
`vec/e_no_q_rep3` and `vec/offline_v2_baseline_r1`. They are kept as recorded;
the cause was not investigated.

## 9. Report figures and PDF

```bash
python scripts/report/make_ablation_figures.py         # outputs/report/figures/fig*.png, summary.json
python scripts/report/diagnose_policy.py --run outputs/ablation/<rung>   # <run>/diagnosis.json
python scripts/report/make_comparison_gif.py --run outputs/ablation/<rung>   # outputs/report/gifs/
python scripts/report/make_pdf_report.py               # outputs/report/comfree_tdmpc_report.pdf (needs `reportlab`)
```

`make_ablation_figures.py` reads `outputs/ablation/` (`--ablation`) and the
benchmark JSONs from section 1; it only needs the CSV/JSON files in the
repository. `diagnose_policy.py` and `make_comparison_gif.py` need `agent.pt`
checkpoints.
