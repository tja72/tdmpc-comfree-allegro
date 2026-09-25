"""Vectorised training entry point.

The three switches below are the three claims of the project, and the ablation
turns them on one at a time:

  --distill          harvest MPPI elite rollouts as extra Q/policy training data
  --sysid            identify the planner's 71 physical parameters online
  --terminal_value   let the planner add gamma^H Q(s_H) to each sampled rollout

  # nothing switched on
  python scripts/train_vec.py --name a_base --steps 200000

  # everything switched on
  python scripts/train_vec.py --name d_full --steps 200000 \
      --distill --sysid --terminal_value
"""

from __future__ import annotations

import argparse

from comfree_tdmpc.agent.tdmpc_agent import AgentConfig
from comfree_tdmpc.envs.task import TaskConfig
from comfree_tdmpc.envs.vec_allegro import VecEnvConfig
from comfree_tdmpc.planner.mppi import PlannerConfig
from comfree_tdmpc.sysid.stochastic import StochasticSysIDConfig
from comfree_tdmpc.vec_trainer import VecTrainConfig, VecTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--tag", type=str, default="", help="label burned into the eval videos")
    ap.add_argument("--steps", type=int, default=200_000, help="total environment steps")
    ap.add_argument("--seed_steps", type=int, default=2_000)
    ap.add_argument("--episode_length", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--updates_per_env_step", type=float, default=1.5)
    ap.add_argument("--max_updates_per_iter", type=int, default=24)

    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--num_elites", type=int, default=32)
    ap.add_argument("--num_pi_trajs", type=int, default=24)

    # the three ablation switches
    ap.add_argument("--distill", action="store_true")
    ap.add_argument("--sysid", action="store_true")
    ap.add_argument("--terminal_value", action="store_true")

    # no planner at all: plain SAC-style actor-critic on the real env
    ap.add_argument("--pure_rl", action="store_true",
                    help="skip the planner entirely; the policy acts in the real "
                         "env directly every step. Mutually exclusive with "
                         "--distill/--sysid/--terminal_value/--dagger_beta/"
                         "--offline_start_val_loss, and forces --actor_mode sac")

    ap.add_argument("--record_topk", type=int, default=8)
    ap.add_argument("--sim_data_ratio", type=float, default=0.5)
    ap.add_argument("--terminal_value_warmup_steps", type=int, default=20_000)
    ap.add_argument("--actor_mode", type=str, default="residual",
                    help="residual | sac | bc | bc_mu | bc_no_q | bc_mu_no_q")
    ap.add_argument("--dagger_beta", type=float, default=0.0,
                    help="per-env probability of executing the policy's own action "
                         "instead of the planner's, with the planner's mu/std kept "
                         "as the label (DAgger-style relabeling); 0.0 = always the planner's")
    ap.add_argument("--prior_coef", type=float, default=0.5)
    ap.add_argument("--mlp_dim", type=int, default=384)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--elite_target", type=str, default="elite",
                    help="what the harvested rollouts hand the policy: elite | mean")
    ap.add_argument("--no_learn_reward", action="store_true")

    ap.add_argument("--reality_perturb", type=float, default=0.25)
    ap.add_argument("--reality_seed", type=int, default=12345)
    ap.add_argument("--sysid_every_steps", type=int, default=5_000)
    ap.add_argument("--sysid_iters", type=int, default=30)
    ap.add_argument("--sysid_warmup_steps", type=int, default=5_000)

    ap.add_argument("--offline_start_val_loss", type=float, default=None,
                    help="SysID val_loss threshold that triggers the offline "
                         "consolidation phase once (unset = feature off)")
    ap.add_argument("--offline_num_updates", type=int, default=50_000)
    ap.add_argument("--offline_batch_size", type=int, default=1024)
    ap.add_argument("--offline_value_real_ratio", type=float, default=0.7)
    ap.add_argument("--offline_confirm_checks", type=int, default=1,
                    help="require val_loss < offline_start_val_loss on this "
                         "many consecutive run_sysid() calls before firing")
    ap.add_argument("--offline_min_step", type=int, default=None,
                    help="hard floor -- never fire before this env step")
    ap.add_argument("--offline_force_step", type=int, default=None,
                    help="ignore val_loss, fire exactly once at this env "
                         "step (takes priority over offline_start_val_loss)")
    ap.add_argument("--offline_mode", type=str, default="once",
                    choices=["once", "periodic"],
                    help="'once' = single consolidation burst (default); "
                         "'periodic' = fire offline_periodic_num_updates "
                         "every time the gate condition is met")
    ap.add_argument("--offline_periodic_num_updates", type=int, default=8_000)

    ap.add_argument("--eval_every_steps", type=int, default=5_000)
    ap.add_argument("--mpc_eval_every_steps", type=int, default=25_000)
    ap.add_argument("--video_every_steps", type=int, default=50_000)
    ap.add_argument("--log_every_steps", type=int, default=2_000)
    ap.add_argument("--out_root", type=str, default="outputs/vec")
    args = ap.parse_args()

    cfg = VecTrainConfig(
        seed=args.seed,
        total_env_steps=args.steps,
        seed_env_steps=args.seed_steps,
        updates_per_env_step=args.updates_per_env_step,
        max_updates_per_iter=args.max_updates_per_iter,
        sim_data_ratio=args.sim_data_ratio if args.distill else 0.0,
        terminal_value_warmup_steps=args.terminal_value_warmup_steps,
        dagger_beta=args.dagger_beta,
        pure_rl=args.pure_rl,
        sysid_enabled=args.sysid,
        sysid_every_steps=args.sysid_every_steps,
        sysid_iters=args.sysid_iters,
        sysid_warmup_steps=args.sysid_warmup_steps,
        offline_start_val_loss=args.offline_start_val_loss,
        offline_num_updates=args.offline_num_updates,
        offline_batch_size=args.offline_batch_size,
        offline_value_real_ratio=args.offline_value_real_ratio,
        offline_confirm_checks=args.offline_confirm_checks,
        offline_min_step=args.offline_min_step,
        offline_force_step=args.offline_force_step,
        offline_mode=args.offline_mode,
        offline_periodic_num_updates=args.offline_periodic_num_updates,
        reality_perturb=args.reality_perturb,
        reality_seed=args.reality_seed,
        eval_every_steps=args.eval_every_steps,
        mpc_eval_every_steps=args.mpc_eval_every_steps,
        video_every_steps=args.video_every_steps,
        log_every_steps=args.log_every_steps,
        out_dir=f"{args.out_root}/{args.name}",
        tag=args.tag or args.name,
        env=VecEnvConfig(
            num_envs=args.num_envs,
            episode_length=args.episode_length,
            task=TaskConfig(),
        ),
        planner=PlannerConfig(
            horizon=args.horizon,
            iterations=args.iterations,
            num_samples=args.num_samples,
            num_elites=args.num_elites,
            num_pi_trajs=args.num_pi_trajs,
            use_terminal_value=args.terminal_value,
            max_std=0.5,
            record_elites=args.distill,
            record_topk=args.record_topk,
            elite_target=args.elite_target,
        ),
        agent=AgentConfig(
            actor_mode=args.actor_mode,
            prior_coef=args.prior_coef,
            learn_reward=not args.no_learn_reward,
            mlp_dim=args.mlp_dim,
            latent_dim=args.latent_dim,
        ),
        sysid=StochasticSysIDConfig(),
    )
    VecTrainer(cfg).train()


if __name__ == "__main__":
    main()
