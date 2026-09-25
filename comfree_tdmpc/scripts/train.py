"""Single-environment training entry point (legacy; see `train_vec.py`).

  # closed loop, reality equal to the compiled model
  python scripts/train.py --name m3_true --episodes 40 --reality_perturb 0

  # same, without the terminal value in the planner
  python scripts/train.py --name m3_noQ --episodes 40 --reality_perturb 0 --no_terminal_value

  # perturbed reality, identified online by SysID
  python scripts/train.py --name m4_sysid --episodes 40

  # control: same perturbed reality, SysID disabled
  python scripts/train.py --name m4_nosysid --episodes 40 --no_sysid
"""

from __future__ import annotations

import argparse

from comfree_tdmpc.agent.tdmpc_agent import AgentConfig
from comfree_tdmpc.envs.allegro import EnvConfig
from comfree_tdmpc.envs.task import TaskConfig
from comfree_tdmpc.planner.mppi import PlannerConfig
from comfree_tdmpc.sysid.stochastic import StochasticSysIDConfig
from comfree_tdmpc.trainer import TrainConfig, Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--tag", type=str, default="", help="label burned into the eval videos")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--episode_length", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed_episodes", type=int, default=2)
    ap.add_argument("--updates_per_step", type=float, default=1.0)

    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--num_pi_trajs", type=int, default=24)
    ap.add_argument("--no_terminal_value", action="store_true")
    ap.add_argument("--distill_rollouts", action="store_true",
                    help="train Q/reward/policy on harvested MPPI elite rollouts too")
    ap.add_argument("--record_topk", type=int, default=8)
    ap.add_argument("--sim_data_ratio", type=float, default=0.5)

    ap.add_argument("--actor_mode", type=str, default="residual")
    ap.add_argument("--no_learn_reward", action="store_true")

    ap.add_argument("--reality_perturb", type=float, default=0.25,
                    help="how far reality sits from the compiled model, in u units")
    ap.add_argument("--reality_seed", type=int, default=12345)
    ap.add_argument("--no_sysid", action="store_true")
    ap.add_argument("--sysid_every_steps", type=int, default=300)
    ap.add_argument("--sysid_iters", type=int, default=12)
    ap.add_argument("--sysid_warmup_steps", type=int, default=240)

    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--eval_episodes", type=int, default=1)
    ap.add_argument("--eval_episodes_pi", type=int, default=3)
    ap.add_argument("--video_every", type=int, default=10)
    ap.add_argument("--out_root", type=str, default="outputs")
    args = ap.parse_args()

    cfg = TrainConfig(
        seed=args.seed,
        total_episodes=args.episodes,
        seed_episodes=args.seed_episodes,
        updates_per_step=args.updates_per_step,
        sim_data_ratio=args.sim_data_ratio if args.distill_rollouts else 0.0,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_episodes_pi=args.eval_episodes_pi,
        video_every=args.video_every,
        sysid_enabled=not args.no_sysid,
        sysid_every_steps=args.sysid_every_steps,
        sysid_iters=args.sysid_iters,
        sysid_warmup_steps=args.sysid_warmup_steps,
        reality_perturb=args.reality_perturb,
        reality_seed=args.reality_seed,
        out_dir=f"{args.out_root}/{args.name}",
        tag=args.tag or args.name,
        env=EnvConfig(episode_length=args.episode_length, task=TaskConfig()),
        planner=PlannerConfig(
            horizon=args.horizon,
            iterations=args.iterations,
            num_samples=args.num_samples,
            num_pi_trajs=args.num_pi_trajs,
            use_terminal_value=not args.no_terminal_value,
            max_std=0.5,
            record_elites=args.distill_rollouts,
            record_topk=args.record_topk,
        ),
        agent=AgentConfig(
            actor_mode=args.actor_mode,
            learn_reward=not args.no_learn_reward,
        ),
        sysid=StochasticSysIDConfig(),
    )
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
