# Experiments

This is an index of the training runs made with this fork, with the command
that reproduces each one. Every run uses `python -m tdmpc_square.train`, started
from `tdmpc-square-pearl/`, and writes to `logs/<task>/<seed>/<exp_name>/`.
Hydra run directories go to `outputs/<date>/<time>/`.

The `logs/` and `outputs/` directories are gitignored and are not part of the
public repository.

Weights & Biases logging:

- Most runs below logged to W&B. To do the same, add
  `disable_wandb=false wandb_entity=<entity>`.
- Without these overrides, runs log only to CSV.

No HumanoidBench, DreamerV3, SAC or PPO runs were made in this fork.

## comfree-allegro-cube (TD-MPC2 baseline for comfree_tdmpc)

Common overrides: `task=comfree-allegro-cube model_size=5 seed=0 eval_value=false eval_pi=true eval_episodes=16`.

| experiment | extra overrides | output dir | notes |
|---|---|---|---|
| smoke_sac | `steps=3000 actor_mode=sac exp_name=smoke_sac eval_freq=1000 save_video=false disable_wandb=true` | `logs/comfree-allegro-cube/0/smoke_sac/` | smoke test |
| smoke_residual | `steps=3000 actor_mode=residual exp_name=smoke_residual eval_freq=1000 save_video=false disable_wandb=true` | `logs/comfree-allegro-cube/0/smoke_residual/` | smoke test |
| residual_1m | `steps=1000000 actor_mode=residual exp_name=residual_1m eval_freq=5000 save_agent_steps=[150000] disable_wandb=false wandb_project=comfree_baseline` | `logs/comfree-allegro-cube/0/residual_1m/` | stopped at about 20k steps, before any checkpoint was saved |

For the full-length TD-MPC2 run, use the `residual_1m` overrides with
`actor_mode=sac exp_name=sac_1m`.

## pusht hyperparameter sweep

Needs `gym-pusht`. Common overrides: `task=pusht seed=0 steps=500000 model_size=5`.
Output dir: `logs/pusht/0/<exp_name>/`, where `<exp_name>` is the experiment
name in the first column.

| experiment | extra overrides | result at ~450k steps |
|---|---|---|
| pusht_test | `horizon=5` | abandoned, no eval |
| pusht_regularized | `horizon=5` | reward ~280; reference checkpoint for the analysis below |
| pusht_regularized_alpha03 | `horizon=5` | reward ~210. No alpha override was applied despite the name, so the default `velocity_penalty_alpha=0.1` was used. |
| pusht_regularized02 | `horizon=5 velocity_penalty_alpha=0.2` | strong until 400k, collapsed at the final eval |
| pusht_delta_actions | `horizon=5 delta_actions=true delta_action_max=50.0` | did not learn |
| pusht_delta_actions_nonRegularized | `horizon=5 delta_actions=true delta_action_max=50.0` | reward ~200. No override was applied despite the name, so the default alpha 0.1 was used. |
| pusht_delta_actions_horizon9 | `horizon=9 delta_actions=true delta_action_max=50.0` | reward ~200 |
| pusht_delta_actions_horizon15_seedSteps20k | `horizon=15 delta_actions=true delta_action_max=50.0 seed_steps=20000` | unstable, collapsed near the end |
| pusht_delta_actions_horizon15_seedSteps20k_minStd01 | same + `min_std=0.1` | reward 267, best of the delta-action runs |
| pusht_delta_actions-max20_nonRegularized | `horizon=5 delta_actions=true delta_action_max=20.0 velocity_penalty_alpha=0.0` | reward 56, slow |
| pusht_deltamax40_horizon15_seedsteps20k_regularized0 | `horizon=15 delta_actions=true delta_action_max=40 seed_steps=20000 velocity_penalty_alpha=0.0` | crashed early |
| pusht_deltamax40_horizon15_FIXEDseedsteps50k_regularized0 | same with `seed_steps=50000` | no eval preserved |
| pusht_deltamax40_horizon9_FIXEDseedsteps50k_regularized0 | `horizon=9 delta_actions=true delta_action_max=40 seed_steps=50000 velocity_penalty_alpha=0.0` | collapsed (0.45) |
| pusht_deltamax40_horizon9_FIXEDseedsteps50k_regularized0_minStd015 | same + `min_std=0.15` | reward 117, still improving |

"FIXED" in a name means the run came after a bug fix. Before the fix, the
automatic `seed_steps` default overwrote an explicit `seed_steps=` override.

Post-hoc analysis was run on the final checkpoints of `pusht_regularized`,
`pusht_regularized_alpha03`, `pusht_delta_actions` and
`pusht_delta_actions_horizon9`. Pass `delta_actions=true` for delta-action
checkpoints.

```bash
python -m evaluate.eval_prediction_loss checkpoint=logs/pusht/0/<exp_name>/models/final.pt task=pusht eval_episodes=10 loss_eval_batches=50
python -m evaluate.visualize_q_pusht    checkpoint=logs/pusht/0/<exp_name>/models/final.pt task=pusht
```

## fingertips-cube-air reward-shaping series

Common overrides: `task=fingertips-cube-air seed=0 steps=2000000 horizon=6 model_size=5`.
Output dir: `logs/fingertips-cube-air/0/<exp_name>/`, where `<exp_name>` is the
experiment name in the first column.

The reward code changed between runs, and only the last version is in this
repository. So only runs that use the current reward can be reproduced
exactly. The current reward version:

- max (not mean) fingertip distance;
- position and quaternion tracking gated on multi-finger contact;
- a 1:1 position:quaternion weight.

| experiment | extra overrides | reward version / result |
|---|---|---|
| fingertips_cube_air | none (goal-conditioned, legacy cost reward) | strongly negative; exploration never reached the object |
| fingertips_cube_air_fixed_goal | `+goal_conditioned=false` | reward ~350 |
| fingertips_cube_air_shaped_fixed_goal | `+goal_conditioned=false +fingertips.reward.style=shaped` | first shaped reward; reward scale not comparable to later runs |
| fingertips_cube_air_shaped_fixGoal_run2 | same | mean to max fingertip distance, rescaled; flat |
| fingertips_cube_air_shaped_fixGoal_run3 | same | contact-gated tracking; converged, stable at 2M |
| fingertips_cube_air_shaped_fixGoal_run3_goalCond | `+goal_conditioned=true +fingertips.reward.style=shaped` | same reward, goal-conditioned; worse than run3 |
| fingertips_cube_air_shaped_fixGoal_run5_original_rew | `+goal_conditioned=false +fingertips.reward.style=legacy +fingertips.reward.legacy.baseline=0 +fingertips.reward.legacy.scale=1` | current code with the legacy reward, as a comparison; negative, worse than shaped |
