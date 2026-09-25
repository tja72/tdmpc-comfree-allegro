"""Why does the distilled policy not match the planner?

Four candidate explanations, and a measurement that separates them:

  1. *Mode averaging.*  MPPI is multi-modal: at one state several unrelated
     action sequences score alike, and a single deterministic policy fitted to
     all of them lands between them, where nothing good happens.  Diagnosed by
     comparing the policy's imitation error against the **spread inside the
     elite set itself** -- the error no deterministic function can beat.
  2. *Mode collapse.*  The policy emits nearly the same action everywhere.
     Diagnosed by the per-dimension standard deviation of its actions over time,
     against the planner's.
  3. *Distribution shift.*  The policy imitates well on states the planner
     visits, then drifts somewhere else when it drives on its own.  Diagnosed by
     measuring the same imitation error on planner-visited states and on
     policy-visited states.
  4. *A bad value function.*  The policy's objective is mostly "maximise Q", so
     an inaccurate Q sends it somewhere the planner never goes.  Diagnosed by
     comparing Q's prediction against the return that actually followed.

    python scripts/report/diagnose_policy.py --run outputs/ablation/d_termq
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.agent.tdmpc_agent import AgentConfig, TDMPCAgent
from comfree_tdmpc.envs.task import TaskConfig
from comfree_tdmpc.envs.vec_allegro import VecAllegroReorientEnv, VecEnvConfig
from comfree_tdmpc.planner.batch_mppi import BatchMPPIPlanner
from comfree_tdmpc.planner.mppi import PlannerConfig
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig, load_spec
from comfree_tdmpc.sim.handle_writer import HandleWriter
from comfree_tdmpc.sim.param_space import ParamSpace


def load(run: Path, num_envs: int, horizon: int, num_samples: int, iterations: int,
         seed: int):
    space = ParamSpace.realistic(load_spec())
    u_model = np.load(run / "params.npy") if (run / "params.npy").exists() else space.true_u
    u_real = (np.load(run / "params_reality.npy")
              if (run / "params_reality.npy").exists() else space.true_u)
    env = VecAllegroReorientEnv(
        VecEnvConfig(num_envs=num_envs, episode_length=120, task=TaskConfig()),
        seed=seed, space=space, u=u_real)
    agent = TDMPCAgent(AgentConfig(), env.obs_dim, env.action_dim, env.torch_device)
    agent.load(str(run / "agent.pt"))
    cfg = PlannerConfig(horizon=horizon, iterations=iterations, num_samples=num_samples,
                        num_pi_trajs=24, use_terminal_value=True, max_std=0.5,
                        record_elites=True, record_topk=8)
    sim = BatchSim(SimConfig(nworld=num_envs * num_samples))
    HandleWriter(sim, space).write_u(u_model)
    planner = BatchMPPIPlanner(sim, env.task, cfg, num_envs, agent=agent)
    return env, agent, planner


@torch.no_grad()
def elite_spread(planner: BatchMPPIPlanner, k: int = 8) -> dict:
    """How far apart are the first actions the planner considers equally good?

    Read straight off the planner's *converged* elite set -- the one the last
    MPPI iteration actually produced, after the sampling distribution has
    narrowed.  Re-sampling it at `max_std` instead would widen the set and
    inflate the floor, which is the direction that would flatter the conclusion.

    If several actions score within a few percent of each other but differ by d
    in action space, no deterministic policy can be closer than about d to all
    of them at once.
    """
    if planner.last_elites is None:
        return {}
    first, value = planner.last_elites  # (E, K, A), (E, K)
    first, value = first[:, :k], value[:, :k]
    centre = first.mean(dim=1, keepdim=True)
    v_hi = value.max(1).values
    v_lo = value.min(1).values
    return {
        # Mean distance of an elite action from the elite centroid, per dimension.
        "mean_abs_dev": float((first - centre).abs().mean()),
        "std": float(first.std(dim=1).mean()),
        # How equal "equally good" really is: the value range inside the top-k,
        # relative to the size of the values themselves.
        "value_range_rel": float(
            ((v_hi - v_lo) / v_hi.abs().clamp(min=1e-6)).mean()),
        "k": float(k),
    }


@torch.no_grad()
def collect(env, agent, planner, steps: int, driver: str) -> dict:
    """Roll out under `driver` and record what each controller would have done."""
    obs = env.reset()
    planner.reset()
    a_plan, a_pi, qs, rewards, alive_mask = [], [], [], [], []
    for t in range(steps):
        qpos, qvel = env.state()
        ap, mu, _ = planner.plan(qpos, qvel, env.goal_quat, env.prev_action,
                                 t0=env.needs_t0.clone(), eval_mode=True)
        api = agent.act_batch(obs, eval_mode=True)
        z = agent.encode(obs)
        qs.append(agent.Q(z, api, return_type="min").squeeze(-1).cpu())
        a_plan.append(mu.cpu())      # the planner's own deterministic action
        a_pi.append(api.cpu())
        obs_next, r, done, info = env.step(ap if driver == "planner" else api)
        rewards.append(r.cpu())
        alive_mask.append((~done).float().cpu())
        obs = env.observation()
    return {
        "a_plan": torch.stack(a_plan),   # (T, E, A)
        "a_pi": torch.stack(a_pi),
        "q": torch.stack(qs),            # (T, E)
        "reward": torch.stack(rewards),
        "alive": torch.stack(alive_mask),
    }


def discounted_return(reward: torch.Tensor, gamma: float, h: int) -> torch.Tensor:
    """Monte-Carlo return over the next `h` steps, for comparing against Q."""
    T = reward.shape[0]
    out = torch.zeros_like(reward)
    for k in range(h):
        out[: T - k] += (gamma ** k) * reward[k:]
    return out[: T - h]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="outputs/ablation/d_termq")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--spread_steps", type=int, default=60,
                    help="states to average the elite spread over")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    run = Path(args.run)
    out = {}
    env, agent, planner = load(run, args.num_envs, args.horizon, args.num_samples,
                               args.iterations, args.seed)

    # 1 & 2: imitation error where the planner drives
    d_plan = collect(env, agent, planner, args.steps, driver="planner")
    d_pi = collect(env, agent, planner, args.steps, driver="policy")

    for tag, d in (("planner_states", d_plan), ("policy_states", d_pi)):
        err = (d["a_pi"] - d["a_plan"]).abs().mean().item()
        out[f"imitation_error/{tag}"] = err
    out["action_std/policy"] = d_plan["a_pi"].std(dim=0).mean().item()
    out["action_std/planner"] = d_plan["a_plan"].std(dim=0).mean().item()
    out["action_absmean/policy"] = d_plan["a_pi"].abs().mean().item()
    out["action_absmean/planner"] = d_plan["a_plan"].abs().mean().item()

    # the irreducible floor: how much do equally good elites disagree?
    # Averaged over many states rather than read off one, since the spread
    # depends on how ambiguous the current state happens to be.
    env.reset()
    planner.reset()
    acc: dict[str, list[float]] = {}
    for t in range(args.spread_steps):
        qpos, qvel = env.state()
        a, _, _ = planner.plan(qpos, qvel, env.goal_quat, env.prev_action,
                               t0=env.needs_t0.clone(), eval_mode=True)
        if t >= 10:  # let the lanes get away from the keyframe first
            for k, v in elite_spread(planner).items():
                acc.setdefault(k, []).append(v)
        env.step(a)
    for k, v in acc.items():
        out[f"elite_spread/{k}"] = float(np.mean(v))

    # 4: is Q telling the truth?
    gamma = agent.cfg.discount
    h = 20
    mc = discounted_return(d_plan["reward"], gamma, h)
    q = d_plan["q"][: mc.shape[0]]
    m = torch.isfinite(mc) & torch.isfinite(q)
    out["value/q_mean"] = float(q[m].mean())
    out["value/mc_return_mean"] = float(mc[m].mean())
    out["value/bias"] = float((q[m] - mc[m]).mean())
    out["value/corr"] = float(np.corrcoef(q[m].numpy(), mc[m].numpy())[0, 1])

    # returns actually achieved
    out["return/planner_driving"] = float(d_plan["reward"].sum(0).mean())
    out["return/policy_driving"] = float(d_pi["reward"].sum(0).mean())

    print(f"\n=== {run.name} ===")
    w = 34
    print(f"{'quantity':<{w}} {'value':>10}   interpretation")
    print(f"{'-'*w} {'-'*10}   {'-'*40}")

    def row(k, v, note=""):
        print(f"{k:<{w}} {v:>10.4f}   {note}")

    row("imitation error, planner states", out["imitation_error/planner_states"],
        "how well pi copies the planner where it looks")
    row("imitation error, policy states", out["imitation_error/policy_states"],
        "the same, where pi itself ends up")
    row("elite spread (mean abs dev)", out["elite_spread/mean_abs_dev"],
        "floor: equally good actions disagree this much")
    row("elite value range (relative)", out["elite_spread/value_range_rel"],
        "how equal 'equally good' really is")
    row("action std, policy", out["action_std/policy"], "low => collapsed to one action")
    row("action std, planner", out["action_std/planner"], "")
    row("Q mean", out["value/q_mean"], "")
    row("Monte-Carlo return mean", out["value/mc_return_mean"], f"over {h} steps")
    row("Q bias", out["value/bias"], "positive => Q is optimistic")
    row("Q vs return correlation", out["value/corr"], "1 = perfect ranking")
    row("return, planner driving", out["return/planner_driving"], "")
    row("return, policy driving", out["return/policy_driving"], "")

    path = Path(args.out or (run / "diagnosis.json"))
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
