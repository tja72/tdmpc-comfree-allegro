# comfree_tdmpc

A TD-M(PC)²-style agent for in-hand cube reorientation with a 16-DoF Allegro
hand, in which the planner's dynamics model is not a learned latent world model
but [ComFree-Sim](https://irislab.tech/comfree-sim/), a GPU-batched
analytic-contact simulator built on MuJoCo-Warp. What is learned is the part of
TD-M(PC)² that carries long-range information: a Q ensemble (used as the
planner's terminal value), a policy (used as a planner prior and as a
standalone controller) and an optional reward head. What replaces world-model
learning is online system identification (SysID) of 71 physical parameters of
the simulator.

The "reality" the agent acts in is the same ComFree simulator with those 71
parameters displaced from their nominal values (fixed seed, rejection-sampled
for stability). The model/reality gap that SysID has to close is therefore
purely parametric; there is no unmodelled physics and no real robot in this
repository.

This directory is one part of the
[tdmpc-comfree-allegro](https://github.com/tja72/tdmpc-comfree-allegro)
monorepo. Installation of the conda environments is described in the root
README; results are discussed in the
[blog post](https://tja72.github.io/tdmpc-comfree-allegro/).

## Method

```
reality: ComFree with perturbed parameters u*
   |  transitions (obs, action, reward, planner mu/std, qpos, qvel)
   v
replay buffer (one ring per environment lane)
   |-- (qpos, qvel, action) chunks --> SysID: batched central finite differences
   |                                   -> identified parameters u_hat
   |-- (obs, action, reward, mu, std) --> TD learning: Q ensemble (two-hot),
   |                                      policy, optional reward head
   v
MPPI over ComFree(u_hat): per environment 256 sampled action sequences
(24 of them from the policy) as worlds of one GPU bank, scored with the
analytic task reward + gamma^H Q(s_H)
   |
   +--> action executed in reality; elite rollouts optionally fed back as
        training data (--distill)
```

- **Planner dynamics are the simulator.** One MPPI iteration for all
  `num_envs` environments is one pass over a bank of ComFree worlds (world
  `e*N + i` is sample `i` of environment `e`), so a horizon-H rollout costs H
  batched control steps regardless of the number of samples. The reward inside
  the planner is the analytic task reward (`envs/task.py`), the same function
  that scores the real environment.
- **State injection is exact up to float32 round-off.** With position
  actuators, a ComFree simulation's state is `(qpos, qvel)`; the buffer stores
  both, so SysID can replay short chunks of real transitions inside the model.
- **SysID uses batched central finite differences, not autodiff.** Per-world
  parameters in MuJoCo-Warp let all perturbed parameter settings times all data
  chunks run as one batched rollout. The 71-parameter optimiser used in
  training (`sysid/stochastic.py`) takes central differences along K=16 random
  Gaussian directions in a normalised coordinate `u in [0, 1]^71`
  (log-uniform for scale parameters, linear for additive ones), with a
  decaying step, momentum and a clip to the unit box. `sysid/fd_optimizer.py`
  is the original coordinate-wise central-difference stencil with a batched
  Armijo line search over a 4-parameter set (`sim/params.py`); it is used by the
  milestone scripts `02_sysid_recovery.py` and `04_check_smoothness.py`.
- **The planner starts from the nominal model.** `u_hat` is initialised at the
  compiled scene's values; reality is `u* = u_nominal + U(-0.25, 0.25)` per
  coordinate (`reality.py`, `reality_seed=12345`, `reality_perturb=0.25`).
  The same `apply_reality()` is used by the TD-MPC2 baseline elsewhere in the
  monorepo so both agents face the same plant.

Nothing is vendored from ComFree-Sim or TD-M(PC)². `comfree_warp` is an
installed dependency; its Allegro scene
(`benchmark/test_data/allegro/env_allegro_cube.xml`) is located through the
installed package (or the `COMFREE_REPO` environment variable) and edited in
memory via `mujoco.MjSpec`. The TD-MPC components (SimNorm encoder, two-hot Q
ensemble, squashed-Gaussian policy, residual policy prior, running scale) are
reimplemented in `comfree_tdmpc/agent/`.

## Package layout

| path | contents |
| --- | --- |
| `comfree_tdmpc/sim/batch_sim.py` | ComFree world bank: per-world parameters, state get/set, CUDA-graph capture, scene loading |
| `comfree_tdmpc/sim/param_space.py` | the 71 SysID parameters (`ParamSpace.realistic`) and their normalised `u` coordinates |
| `comfree_tdmpc/sim/handle_writer.py` | writes a `u` vector into every world of a bank in one pass |
| `comfree_tdmpc/sim/params.py` | the original 4-parameter set (used by milestone scripts 00-04 and legacy report scripts) |
| `comfree_tdmpc/reality.py` | draws and applies the perturbed "reality" parameters |
| `comfree_tdmpc/envs/task.py` | observation, analytic reward and goal logic, batched in torch |
| `comfree_tdmpc/envs/allegro.py` | single environment |
| `comfree_tdmpc/envs/vec_allegro.py` | `num_envs` environments in lockstep with per-lane auto-reset |
| `comfree_tdmpc/planner/mppi.py` | MPPI with simulator dynamics and learned terminal value (single env) |
| `comfree_tdmpc/planner/batch_mppi.py` | the same planner for all environments at once; harvests elite rollouts for distillation |
| `comfree_tdmpc/agent/nets.py`, `tdmpc_agent.py` | networks and TD learning of Q, policy and reward head |
| `comfree_tdmpc/agent/buffer.py`, `vec_buffer.py` | GPU replay buffers storing `(qpos, qvel)` alongside the usual transition fields |
| `comfree_tdmpc/sysid/stochastic.py` | 71-parameter SysID (random-direction central differences) used in training |
| `comfree_tdmpc/sysid/fd_optimizer.py` | 4-parameter SysID (coordinate-wise central differences + Armijo line search) |
| `comfree_tdmpc/vec_trainer.py` | the closed loop for `num_envs` environments, step-budgeted; optional offline phase |
| `comfree_tdmpc/trainer.py` | legacy single-environment closed loop |
| `scripts/00_*.py` ... `scripts/08_*.py` | milestone checks and benchmarks |
| `scripts/train_vec.py` | training entry point |
| `scripts/run_ablation.py` | runs the A-F ablation ladder via `train_vec.py` |
| `scripts/report/` | figures, diagnostics, gifs, PDF report |
| `docs/offline_training_design.md` | design of the offline consolidation phase (`--offline_*` flags) |

## Installation

See the root README of the monorepo for the conda environment. In short, this
package needs a CUDA GPU visible as `cuda:0` (the device is not configurable and
there is no CPU fallback), an editable install of `comfree_warp`, and an
editable install of this package (`pip install -e .`).

`warp-lang` must be pinned to exactly **1.14.0**. ComFree's vendored
`mujoco_warp/_src/sensor.py` contains a latent bug (`_frame_axis` reads `xmat`
in a branch that never defines it). warp 1.14.0 compiles it; warp 1.16 refuses
with `WarpCodegenKeyError: Referencing undefined symbol: xmat` when the first
simulator is built. `comfree_warp` itself only requires `warp-lang>=1.12`, so
install the pin explicitly:

```bash
pip install "warp-lang==1.14.0"
```

`scripts/report/make_pdf_report.py` additionally needs `reportlab` (included in
`envs/comfree-tdmpc.yml`, or `pip install -e ".[report]"`).

## Usage

All commands are run from `comfree_tdmpc/`. Set `MUJOCO_GL=egl` for anything
that renders (the smoke test, training runs record evaluation videos, gif
scripts).

### Smoke test

```bash
MUJOCO_GL=egl python scripts/00_smoke_test.py --nworlds 1 4
```

Steps the ComFree bank, checks that per-world parameters take effect and that
state injection is exact, and prints a throughput table. The first run compiles
the Warp kernels (about a minute); they are cached afterwards.

### Training

`scripts/train_vec.py` is the training entry point. The three main switches
are off by default:

- `--distill` harvests MPPI elite rollouts as additional Q/policy training data
- `--sysid` identifies the planner's 71 parameters online
- `--terminal_value` adds `gamma^H Q(s_H)` to every sampled rollout

Rung D of the ablation (`outputs/ablation/d_termq/config.json`):

```bash
MUJOCO_GL=egl python scripts/train_vec.py --name d_termq --out_root outputs/ablation \
    --steps 150000 --num_envs 16 --horizon 8 --iterations 3 --seed 0 \
    --distill --sysid --terminal_value
```

`--num_envs 16 --horizon 8 --iterations 3 --seed 0` are the defaults and can be
omitted. Other relevant flags: `--actor_mode {residual,sac,bc,bc_mu,bc_no_q,bc_mu_no_q}`,
`--elite_target {elite,mean}`, `--sim_data_ratio`, `--prior_coef`,
`--dagger_beta`, `--num_samples`, `--mlp_dim`, `--latent_dim`, `--pure_rl`
(no planner) and the `--offline_*` flags. See `python scripts/train_vec.py --help`.
The run directory is `<out_root>/<name>/` (default `out_root` is `outputs/vec`).

### Ablation ladder

```bash
MUJOCO_GL=egl python scripts/run_ablation.py --steps 150000   # all rungs a_base ... f_meantarget
python scripts/run_ablation.py --list                          # rungs, their flags, and which are done
```

Runs are sequential (16 environments plus the planner fill one GPU). Rungs
whose `config.json` already exists are skipped unless `--force` is given;
`--only <rung> ...` and `--out_root` select a subset and an output directory.

### Figures and report

```bash
python scripts/report/make_ablation_figures.py                    # outputs/report/figures/*.png + summary.json
python scripts/report/diagnose_policy.py --run outputs/ablation/d_termq   # writes <run>/diagnosis.json
MUJOCO_GL=egl python scripts/report/make_comparison_gif.py --run outputs/ablation/d_termq
python scripts/report/ablate_rollout_budget.py                    # outputs/rollout_budget/
python scripts/report/make_pdf_report.py                          # outputs/report/comfree_tdmpc_report.pdf
```

`make_ablation_figures.py` and `make_pdf_report.py` read `outputs/ablation/`
and the benchmark JSONs in `outputs/06_throughput/`, `07_verify/` and
`08_tune/`. `diagnose_policy.py` builds a default `AgentConfig` rather than
reading the run's `config.json["agent"]`: checkpoints with a non-default
network size do not load, and checkpoints trained with a non-default
`actor_mode` load but are analysed as if trained with the default one.

### Legacy single-environment path

`scripts/train.py` and `comfree_tdmpc/trainer.py` are the single-environment
reference implementation. They are kept as the correctness reference that
`scripts/07_verify_vec.py` checks the vectorised stack against. The report
scripts `make_figures.py`, `make_demo_gifs.py`, `make_model_comparison.py`,
`ablate_controllers.py` and `policy_baselines.py`, and
`scripts/run_experiments.py`, belong to this path; their default `--run(s)`
directories are not part of the repository.

### Full reproduction

[EXPERIMENTS.md](EXPERIMENTS.md) lists every experiment in `outputs/` with the
exact command and output directory.

## Outputs

Each training run directory contains `config.json` (the full resolved
configuration), `train.csv`, `eval.csv`, `sysid.csv`, `params.npy` (final
identified `u`), `params_reality.npy` (reality `u*`) and, where the diagnostic
was run, `diagnosis.json`. Checkpoints (`agent.pt`), evaluation videos and raw
console logs are not included in the repository.

## Results: ablation ladder

Six runs of 150k environment steps each (`scripts/run_ablation.py`, seed 0,
16 environments, horizon 8, 3 MPPI iterations, 256 samples). The ladder is
cumulative: each rung keeps the previous rung's flags and adds its own.

| rung | flags added | planner goals/ep | planner drop rate | policy goals/ep |
| --- | --- | --- | --- | --- |
| A `a_base` | none | 1.62 ± 0.05 | 0.75 | 0.57 ± 0.16 |
| B `b_distill` | `--distill` | 2.02 ± 0.13 | 0.69 | 1.14 ± 0.36 |
| C `c_sysid` | `--sysid` | 1.94 ± 0.15 | 0.88 | 1.25 ± 0.32 |
| D `d_termq` | `--terminal_value` | 4.56 ± 0.36 | 0.02 | 0.65 ± 0.13 |
| E `e_bc` | `--actor_mode bc --sim_data_ratio 0.75` | 4.69 ± 0.53 | 0.06 | 0.65 ± 0.39 |
| F `f_meantarget` | `--actor_mode bc_mu --elite_target mean --sim_data_ratio 0.75 --prior_coef 2.0` (on top of D) | 4.40 ± 0.58 | 0.04 | 0.93 ± 0.24 |

Values are from `outputs/report/figures/summary.json` for the first repetition
(`outputs/ablation/`). Planner numbers are the mean ± std over the last 3
planner evaluations, policy numbers over the last 6 policy evaluations; each
evaluation is 16 episodes of 120 control steps. Goals/ep counts goal
orientations reached per episode; drop rate is the fraction of episodes in
which the cube falls.

Observations from these runs:

- **The terminal value is the largest effect.** Adding it takes the planner's
  drop rate from 0.88 (C) to 0.02 (D) and more than doubles goals per episode.
  Dropping costs a large penalty, but a horizon-8 planner only sees it once the
  cube is already sliding; Q scores the state earlier.
- **SysID mostly corrects the contact model.** In `sysid.csv`, between the
  first and the last SysID call, the mean error `|u_hat - u*|` of the two
  ComFree contact constants (`err/comfree`) goes from 0.19 to 0.09 (C) and from
  0.19 to 0.11 (D); the joint group barely moves (0.085 to 0.079 in C) and the
  inertial group gets worse (0.092 to 0.107 in C).
- **The policy does not catch up with the planner.** In `diagnosis.json`, the
  spread within the planner's own elite set is 0.11-0.14, while the policy's
  imitation error on planner-visited states is 0.19-0.51. The ratio of
  imitation error on policy-visited to planner-visited states grows down the
  ladder: 1.05, 1.07, 1.25, 2.04, 2.35 for A-E. The best imitator (E) is not the
  best controller.

Caveats: the table is a single repetition; two further repetitions with the
same configuration are in `outputs/ablation_rep2/` and `outputs/ablation_rep3/`
(the GPU simulator is not bitwise deterministic, so same-seed reruns differ).
None of the runs has converged at 150k steps. The reality gap is parametric by
construction.

## SysID parameters

`ParamSpace.realistic()` in `comfree_tdmpc/sim/param_space.py` defines 71
parameters. Scale parameters are multiplicative factors on the compiled model's
value with log-uniform `u`; additive parameters (nominally zero in the XML)
are linear in `u`.

| group | count | parameters | range |
| --- | --- | --- | --- |
| joint | 48 | per DoF (16): damping (scale); armature (additive); dry friction loss (additive) | damping 0.3-3x; armature 0-2e-3; friction loss 0-1e-2 |
| actuator | 8 | position gain `kp` and velocity gain `kv` per joint index within a finger (4 each), shared across the four fingers; `kp` is written to `gainprm[0]` and, negated, to `biasprm[1]` | 0.5-2x |
| contact | 7 | sliding friction per hand geom group (index, middle, ring, thumb, palm); cube sliding friction; cube torsional friction | 0.3-3x |
| inertial | 6 | link mass per finger (4; inertia scaled with mass); cube mass; cube inertia (one isotropic scale) | finger 0.7-1.4x; cube mass 0.4-2.5x; cube inertia 0.5-2x |
| comfree | 2 | ComFree contact stiffness (nominal 0.1) and damping (nominal 1e-3) | 0.3-3x |

## License

The code in this directory is released under the MIT license (see the root
`LICENSE` of the monorepo). It requires `comfree_core` from ComFree-Sim, which
is distributed under the noncommercial ComFree Core Academic Research License;
any use of this code together with `comfree_core` is subject to that license.
