# sim2sim: ComFree-Sim to CPU MuJoCo evaluation

This tool evaluates agents trained on the Allegro cube-reorientation task in
ComFree-Sim twice: once in ComFree-Sim and once in stock CPU MuJoCo
(`mujoco.mj_step`, whose contacts use a soft-constraint complementarity model).
It supports these agents:

- `comfree_tdmpc` agents, run policy-only (`comfree_pi`) or with the ComFree
  MPPI planner (`comfree_mpc`).
- `tdmpc-square-pearl` TD-MPC2 agents trained on `task=comfree-allegro-cube`,
  run with the planner (`pearl_mpc`) or policy-only (`pearl_pi`).

Both backends use the same scene, the same physical parameters ("reality") and
the same task code. The contact solver is the only intended difference, so the
drop in performance from ComFree to MuJoCo estimates the gap caused by the
contact model.

Results: see the blog post.

## Requirements

- The `tdmpc-baseline` conda env (installation: root README). `evaluate.py`
  asserts `mujoco==3.6.0` and `warp-lang==1.14.0`.
- A CUDA GPU at `cuda:0`. The ComFree reference env, the planner's sample bank
  and the task tensors all run on the GPU, including during MuJoCo evaluation.
- Trained checkpoints. `*.pt` files are not in the repository. They are
  distributed as GitHub Release assets, and each one has to go into its run
  directory:
  - `comfree_tdmpc`: `comfree_tdmpc/outputs/<...>/<run>/agent.pt`, next to the
    tracked `config.json`, `eval.csv`, `params.npy` and `params_reality.npy`.
  - pearl: `tdmpc-square-pearl/logs/comfree-allegro-cube/<seed>/<exp_name>/models/<ckpt>.pt`
    (`final.pt` by default, or e.g. `150000.pt`). Pearl agents also need their
    Hydra run directory with `.hydra/config.yaml`. It is looked up
    automatically under `tdmpc-square-pearl/outputs/*/*/` by matching the
    `task`, `seed` and `exp_name` overrides. If that fails, pass it with
    `--pearl_hydra_dir`.

## How the setup maps onto MuJoCo

- **Model.** Both backends use `comfree_tdmpc.sim.batch_sim.load_spec()`: the
  same XML and `MjSpec` edits, timestep 0.002 s, 10 substeps (50 Hz control)
  and `ccd_iterations=50`.
- **Reality parameters.** The reality u-vector comes from
  `comfree_tdmpc.reality.apply_reality` (default seed 12345, perturbation
  0.25). It is always computed on ComFree and checked against the run's
  `params_reality.npy`. `sim2sim/params_cpu.py:CpuHandleWriter` writes it into
  the CPU `MjModel` through the same `ParamSpace` handles, including aliases,
  in the same order as ComFree's `HandleWriter`, then calls `mj_setConst`.
  - The mass/inertia pass for the handles' `extra_bodies` runs last, as it does
    in ComFree. As a result, the cube's inertia follows `cube/mass` on both
    backends.
  - The two ComFree contact parameters (`comfree_stiffness` and
    `comfree_damping`) have no MuJoCo equivalent. They are skipped, and the
    skip is logged.
- **Task code.** The unchanged `VecAllegroReorientEnv` from `comfree_tdmpc`
  provides observations, reward, goals and drop detection. The only change is
  that its sim object is replaced by `MujocoBatchSim`
  (`sim2sim/mujoco_backend.py`), which implements the `BatchSim` interface on
  top of one CPU `MjData` per lane.
  - MuJoCo keeps its state in float64.
  - The task-facing tensors are float32 on the GPU, so goals are drawn from the
    same CUDA RNG stream as in ComFree.
  - Pearl agents run through pearl's own gym wrapper, with the same sim
    replacement.
- **Planner agents.** The MPPI planner keeps planning in its own ComFree GPU
  bank, which holds the run's `params.npy`: the end-of-training SysID estimate,
  or the nominal model if SysID was off. At every control step it receives the
  MuJoCo state (qpos, qvel). That is ComFree's full state, so the planner is
  exactly synchronized with the plant. The planner gets no information about
  the MuJoCo plant.

## Solver configs (`sim2sim/solver_configs.py`)

`default` is the scene as compiled, and it gives the headline number:

- Newton solver and a pyramidal friction cone.
- MuJoCo's default `solref` (0.02, 1) and `solimp` (0.9, 0.95, 0.001).
- No noslip iterations.

The sweep changes one setting at a time and does not calibrate MuJoCo towards
ComFree. `solref`/`solimp` changes apply to contacts only (`geom_solref`,
`geom_solimp`). Joint-limit and equality constraints keep their compiled
values.

| config | change from `default` |
|---|---|
| `cone_elliptic` | elliptic friction cone |
| `solref_tc0.01` | contact time constant 0.01 (stiffer) |
| `solref_tc0.04` | contact time constant 0.04 (softer) |
| `solimp_hard` | solimp (0.95, 0.99, 0.001) |
| `solimp_soft` | solimp (0.8, 0.9, 0.001) |
| `noslip10` | 10 noslip iterations |

## Scripts

Run all commands from the monorepo root with the `tdmpc-baseline` env
activated. Each script adds `sim2sim/` to `sys.path` itself. Relative `--run`
paths are resolved against the current working directory. The scripts set
`MUJOCO_GL=egl` if it is unset.

Avoid ad-hoc `python -c` imports from the monorepo root. There, the
`comfree_warp/` source directory shadows the installed package as a namespace
package. The scripts are not affected.

### `check_parity.py`: model parity and keyframe hold

For each solver config, this script checks two things:

- Every model field written by the parameter handles matches world 0 of a
  ComFree env with the same u. The whole array is compared, and float32
  rounding gives a max relative error of about 1e-7.
- The initial keyframe stays stable when held.

With `--run`, it also asserts that u equals that run's `params_reality.npy`.
It exits non-zero on failure.

```bash
python sim2sim/scripts/check_parity.py --run comfree_tdmpc/outputs/ablation/e_bc --solver all
```

Flags: `--run`, `--reality_seed` (default 12345), `--reality_perturb`
(default 0.25), `--solver NAME|all` (default `default`), `--out FILE.json`.

### `evaluate.py`: closed-loop evaluation in ComFree and MuJoCo

This script runs `--episodes` episodes in ComFree, which serves as the
reference and as a self-check against the run's own `eval.csv`. It then runs
the same episodes in MuJoCo for each selected solver config and reports the
gap between the two.

```bash
python sim2sim/scripts/evaluate.py --agent comfree_mpc --run comfree_tdmpc/outputs/ablation/e_bc --episodes 64
```

Other agent types:

```bash
python sim2sim/scripts/evaluate.py --agent comfree_pi --run comfree_tdmpc/outputs/ablation/e_bc --episodes 64 --sweep --video 4
python sim2sim/scripts/evaluate.py --agent pearl_mpc  --run tdmpc-square-pearl/logs/comfree-allegro-cube/0/<exp_name> --episodes 64
python sim2sim/scripts/evaluate.py --agent pearl_pi   --run tdmpc-square-pearl/logs/comfree-allegro-cube/0/<exp_name> --pearl_ckpt 150000.pt
```

| flag | meaning |
|---|---|
| `--agent` | `comfree_pi`, `comfree_mpc`, `pearl_mpc`, `pearl_pi` |
| `--run` | run directory (see Requirements) |
| `--episodes`, `--seed` | number of episodes, and the evaluation seed |
| `--num_envs` | parallel lanes for comfree agents; pearl always uses 1 |
| `--solver NAME` / `--sweep` | one solver config (default `default`), or all of them |
| `--reality_seed`, `--reality_perturb` | override the reality (comfree agents only; by default they come from the run's config) |
| `--num_samples` | override the planner's sample count (`comfree_mpc`) |
| `--video K` | GIFs of the first K MuJoCo episodes |
| `--skip_comfree` | reuse an existing `<tag>/comfree/episodes.csv` |
| `--no_comfree` | MuJoCo only, with no reference and no gap |
| `--pearl_ckpt` | checkpoint file in `<run>/models/` (default `final.pt`) |
| `--pearl_hydra_dir` | Hydra run dir, if automatic lookup fails |
| `--out`, `--tag` | output root (default `sim2sim/outputs`) and subdirectory name |

### `open_loop_gap.py`: open-loop trajectory gap

This script records action sequences from agent rollouts in ComFree. It then
replays each sequence open-loop from the rollout's initial state:

- twice in ComFree, which gives the noise floor, because the GPU sim is not
  bitwise deterministic;
- once in MuJoCo for each solver config.

It reports per-step cube position error, cube orientation error and hand-qpos
RMS error against the first ComFree replay. Steps after the source episode
ended are masked out. The result does not depend on closed-loop policy
behavior.

```bash
python sim2sim/scripts/open_loop_gap.py --agent comfree_pi --run comfree_tdmpc/outputs/ablation/e_bc --episodes 64 --sweep
```

Flags: `--agent comfree_pi|comfree_mpc`, `--run` (required), `--episodes`,
`--num_envs` (default 16), `--seed`, `--solver` / `--sweep`, `--out`, `--tag`.

### `run_all.py`: full evaluation batch and summary tables

This script runs every evaluation in stages and then aggregates the results
into `sim2sim/outputs/summary_table.{csv,md}`:

| stage | what it runs |
|---|---|
| A | parity checks |
| B | policy-only runs for every `comfree_tdmpc` run with a checkpoint |
| C | MPC on the core set of runs |
| D | solver sweeps |
| E | open-loop gap |
| F | pearl checkpoints, when present |
| G | aggregation into the summary tables |

- **Discovery.** It finds runs by scanning for `agent.pt` under
  `comfree_tdmpc/outputs/`, and for pearl checkpoints under
  `tdmpc-square-pearl/logs/comfree-allegro-cube/`. Runs without a checkpoint
  are skipped.
- **Resuming.** A job whose `summary.json` already exists is skipped, so the
  script can be killed and restarted. Failed jobs are logged and skipped.
- **Runtime.** On one RTX 4090 the default `--mpc core` set takes roughly
  3 h, and `--mpc all` roughly 6 h.

```bash
mkdir -p sim2sim/outputs
nohup setsid python sim2sim/scripts/run_all.py > sim2sim/outputs/run_all.log 2>&1 &
```

| flag | meaning |
|---|---|
| `--mpc core\|all\|none` | which runs get MPC evaluation |
| `--stages` | subset of stages, e.g. `BC` (default `ABCDEF`; aggregation always runs) |
| `--dry_run` | print the plan and the time estimate only |
| `--aggregate_only` | rebuild the summary tables from existing outputs |

## Outputs

Only `sim2sim/outputs/summary_table.csv` and `sim2sim/outputs/summary_table.md`
are tracked in git. Everything else under `sim2sim/outputs/` is gitignored and
is regenerated by the scripts.

The default tag is `<agent>__<run path relative to comfree_tdmpc/outputs or
tdmpc-square-pearl/logs, with / replaced by _>`. Under
`sim2sim/outputs/<tag>/`:

- `comfree/`
  - `episodes.csv`
  - `summary.json`: mean, std and bootstrap 95% CI per metric, plus
    `self_check` against the run's final `eval.csv` row
  - `config.json`
- `<solver>/`
  - `episodes.csv`
  - `summary.json`: the `mujoco` and `comfree` statistics, plus
    `gap_mujoco_minus_comfree` with a bootstrap CI
  - `config.json`: solver settings, compiled solver options, skipped
    parameters, parity report, keyframe hold, package versions
  - `videos/ep*.gif`, when `--video` is set

Under `sim2sim/outputs/openloop__<tag>/`:

- `open_loop_gap.csv`: per-step median, IQR and mean of each error
- `summary.json`
- `open_loop_gap.png`

`episodes.csv` has these columns:

| column | meaning |
|---|---|
| `return` | episode return |
| `goals_reached` | number of goals solved |
| `success` | at least one goal solved |
| `time_to_success` | steps and seconds to the first solved goal |
| `dropped` | the cube was dropped |
| `length` | episode length |
| `bad_qacc_resets` | MuJoCo divergence resets |
| `nonfinite_action_steps` | steps with non-finite actions |

## Statistics and caveats

- **Episode rounds.** Episodes run in rounds of `num_envs` lanes, the same way
  as `comfree_tdmpc`'s `VecTrainer.evaluate()`: one episode per lane, and each
  lane is masked once its episode ends.
- **Goal pairing.** The goal RNG is reseeded per round from `(seed, round)`, so
  ComFree and MuJoCo start each round from the same goals and reset noise.
  After the first solved goal the goal sequences can diverge, and that is part
  of the gap.
- **Gap CI.** The gap CI is an unpaired bootstrap of the difference of means.
- **Self-check.** The ComFree self-check is a z-test against the run's last
  `eval.csv` row, which holds 16 episodes with unseeded goals. It is not an
  exact match. It reports `null` when that row's step does not match the
  checkpoint's step.
- **Diverged checkpoints.** A checkpoint whose policy outputs non-finite
  actions yields a meaningless gap. ComFree turns NaN controls into a NaN state,
  which counts as a drop. MuJoCo rejects the controls. The harness reports such
  runs through `nonfinite_action_steps` and prints a warning.
- **GJK warnings.** `Warning: opt.ccd_iterations ... needs to be increased`
  messages come from ComFree's GPU GJK, not from CPU MuJoCo. They also appear
  during training.
