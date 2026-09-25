"""Episode loops and statistics.

`run_vec_episodes` is `VecTrainer.evaluate()` (comfree_tdmpc/vec_trainer.py)
repeated over rounds: one episode per lane, lanes that terminate keep
stepping (auto-reset) but are masked out, goals/dropped taken at the first
`done`, lanes that never terminate credited with their running goal count.
`run_gym_episodes` is pearl's `OnlineTrainer.eval` loop on pearl's own gym
wrapper, one lane, episodes in sequence.

Pairing: before each round / episode the env's goal RNG is reseeded from
(seed, round), so the ComFree and MuJoCo runs start every round from the
same goals and reset noise (both envs keep that RNG on cuda).  After the
first goal is solved the streams may diverge -- that is part of the gap.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch

EP_FIELDS = ["episode", "round", "lane", "return", "goals_reached", "success",
             "time_to_success", "time_to_success_s", "dropped", "length", "bad_qacc_resets", "nonfinite_action_steps"]


def _round_seed(seed: int, r: int) -> int:
    return 1_000_003 * (seed + 1) + r


def _bad_qacc(env) -> np.ndarray | None:
    """Per-lane count of MuJoCo's automatic resets on bad qacc; None on ComFree."""
    b = getattr(env.sim, "bad_qacc", None)
    return None if b is None else b.copy()


@torch.no_grad()
def run_vec_episodes(env, controller, episodes: int, seed: int = 0,
                     frame_fn=None, record_rollouts: bool = False) -> tuple[list[dict], list[dict]]:
    """Returns (per-episode records, rollouts).  rollouts only if record_rollouts:
    one dict per round with qpos0/qvel0/ctrl0 (E, .), actions (T, E, A), live (T, E)."""
    E, T = env.num_envs, env.cfg.episode_length
    dev = env.torch_device
    dt = env.control_dt
    records, rollouts = [], []
    torch.manual_seed(seed)
    for r in range(math.ceil(episodes / E)):
        env.rng.manual_seed(_round_seed(seed, r))
        env.reset()
        controller.reset()
        bad0 = _bad_qacc(env)
        roll = None
        if record_rollouts:
            q, v = env.state()
            roll = {"qpos0": q.cpu().numpy(), "qvel0": v.cpu().numpy(),
                    "ctrl0": env.sim.ctrl.detach().cpu().numpy().copy(), "actions": [], "live": []}

        returns = torch.zeros(E, device=dev)
        goals = torch.zeros(E, device=dev)
        drops = torch.zeros(E, device=dev)
        lengths = torch.zeros(E, device=dev)
        first_goal = torch.full((E,), -1, dtype=torch.long, device=dev)
        nonfinite = torch.zeros(E, dtype=torch.long, device=dev)
        live = torch.ones(E, dtype=torch.bool, device=dev)
        for t in range(T):
            a = controller.act(env)
            # A diverged agent emits NaN actions; the two sims handle NaN ctrl differently.
            nonfinite += (live & ~torch.isfinite(a).all(dim=-1)).long()
            if roll is not None:
                roll["actions"].append(a.detach().float().cpu().numpy())
                roll["live"].append(live.cpu().numpy())
            _, rew, done, info = env.step(a)
            returns += rew * live.float()
            lengths += live.float()
            hit = live & (info["goals_reached"] > 0) & (first_goal < 0)
            first_goal = torch.where(hit, torch.full_like(first_goal, t + 1), first_goal)
            ended = live & done
            goals = torch.where(ended, info["goals_reached"].float(), goals)
            drops = torch.where(ended, info["dropped"], drops)
            if frame_fn is not None and r == 0:
                frame_fn(env, t, info, live & ~done)
            live = live & ~done
            if not bool(live.any()):
                break
        goals = torch.where(live, env.goals_reached.float(), goals)
        bad = (_bad_qacc(env) - bad0) if bad0 is not None else np.zeros(E, dtype=np.int64)
        if roll is not None:
            roll["actions"] = np.stack(roll["actions"])
            roll["live"] = np.stack(roll["live"])
            rollouts.append(roll)

        for i in range(E):
            n = len(records)
            if n >= episodes:
                break
            fg = int(first_goal[i])
            records.append({
                "episode": n, "round": r, "lane": i,
                "return": float(returns[i]), "goals_reached": float(goals[i]),
                "success": float(goals[i] >= 1),
                "time_to_success": fg if fg > 0 else float("nan"),
                "time_to_success_s": fg * dt if fg > 0 else float("nan"),
                "dropped": float(drops[i]), "length": float(lengths[i]),
                "bad_qacc_resets": int(bad[i]), "nonfinite_action_steps": int(nonfinite[i]),
            })
    return records, rollouts


@torch.no_grad()
def run_gym_episodes(gym_env, agent, use_pi: bool, episodes: int, seed: int = 0,
                     frame_fn=None) -> list[dict]:
    """pearl's eval loop: `agent.act(obs, t0=t==0, eval_mode=True, use_pi=...)`."""
    vec = gym_env.env.env  # TensorWrapper -> ComFreeAllegroEnv -> VecAllegroReorientEnv
    dt = vec.control_dt
    records = []
    torch.manual_seed(seed)
    for ep in range(episodes):
        vec.rng.manual_seed(_round_seed(seed, ep))
        bad0 = _bad_qacc(vec)
        obs, _ = gym_env.reset()
        done, t, ret, first, info, nonfinite = False, 0, 0.0, -1, {}, 0
        while not done:
            a = agent.act(obs, t0=t == 0, eval_mode=True, use_pi=use_pi)[0]
            nonfinite += int(not bool(torch.isfinite(a).all()))
            obs, reward, terminated, truncated, info = gym_env.step(a)
            ret += float(reward)
            t += 1
            if first < 0 and info["goals_reached"] >= 1:
                first = t
            if frame_fn is not None and ep == 0:
                frame_fn(vec, t - 1, info, None)
            done = bool(terminated) or bool(truncated)
        epi = info.get("episode", {})
        bad = (_bad_qacc(vec) - bad0) if bad0 is not None else np.zeros(1, dtype=np.int64)
        records.append({
            "episode": ep, "round": ep, "lane": 0,
            "return": float(epi.get("episode_reward", ret)),
            "goals_reached": float(epi.get("goals_reached", info["goals_reached"])),
            "success": float(epi.get("goals_reached", info["goals_reached"]) >= 1),
            "time_to_success": first if first > 0 else float("nan"),
            "time_to_success_s": first * dt if first > 0 else float("nan"),
            "dropped": float(epi.get("dropped", info["dropped"])),
            "length": float(epi.get("episode_length", t)),
            "bad_qacc_resets": int(bad[0]), "nonfinite_action_steps": nonfinite,
        })
    return records


def write_episodes(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EP_FIELDS)
        w.writeheader()
        w.writerows(records)


def read_episodes(path: Path) -> list[dict]:
    with open(path) as f:
        return [{k: float(v) for k, v in row.items() if v not in ("", None)} for row in csv.DictReader(f)]


def _boot_ci(x: np.ndarray, rng, n_boot: int, stat=np.mean) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    s = stat(x[idx], axis=1)
    return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


METRICS = ["return", "goals_reached", "success", "dropped", "length", "time_to_success_s"]


def summarize(records: list[dict], n_boot: int = 10_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {"n": len(records)}
    for k in METRICS:
        x = np.array([r[k] for r in records], dtype=np.float64)
        if k == "time_to_success_s":  # only over episodes that solved a goal
            x = x[np.isfinite(x)]
        if len(x) == 0:
            out[k] = {"mean": float("nan"), "std": float("nan"), "ci95": [float("nan")] * 2, "n": 0}
            continue
        out[k] = {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                  "ci95": list(_boot_ci(x, rng, n_boot)), "n": int(len(x))}
    out["bad_qacc_resets_total"] = int(sum(r.get("bad_qacc_resets", 0) for r in records))
    out["nonfinite_action_steps_total"] = int(sum(r.get("nonfinite_action_steps", 0) for r in records))
    return out


def gap(ref: list[dict], other: list[dict], n_boot: int = 10_000, seed: int = 0) -> dict:
    """other - ref, with an (unpaired) bootstrap CI on the difference of means."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in METRICS:
        a = np.array([r[k] for r in ref], dtype=np.float64)
        b = np.array([r[k] for r in other], dtype=np.float64)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            out[k] = {"diff": float("nan"), "ci95": [float("nan")] * 2}
            continue
        ia = rng.integers(0, len(a), (n_boot, len(a)))
        ib = rng.integers(0, len(b), (n_boot, len(b)))
        d = b[ib].mean(1) - a[ia].mean(1)
        rel = (b.mean() - a.mean()) / abs(a.mean()) if a.mean() != 0 else float("nan")
        out[k] = {"diff": float(b.mean() - a.mean()), "rel": float(rel),
                  "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}
    return out


def self_check(summary: dict, eval_csv: Path, prefix: str, z_tol: float = 2.5,
               ckpt_step: int | None = None) -> dict:
    """Harness(ComFree) vs the run's own final eval.csv row (16 episodes, unseeded goals).

    Per metric: z = |mean_h - mean_csv| / sqrt(sd_h^2/n_h + sd_csv^2/16).  eval.csv only
    logs a std for the return; for `dropped` both sides use the pooled Bernoulli sd, for
    `goals_reached` the harness sd floored at the pooled Poisson sd.  `length` is reported
    but not tested (no usable spread estimate; it is ~120 for any agent that rarely drops).
    """
    rows = list(csv.DictReader(open(eval_csv)))
    row = next((r for r in reversed(rows) if r.get(f"{prefix}/return", "") not in ("", None)), None)
    if row is None:
        return {"available": False, "reason": f"no {prefix}/* rows in {eval_csv}"}
    n_csv = 16
    out = {"available": True, "eval_csv_step": int(float(row["step"])), "prefix": prefix, "metrics": {}}
    if ckpt_step is not None:
        out["ckpt_step"] = int(ckpt_step)
        # The final eval row must come from the same agent state as the checkpoint.
        out["comparable"] = bool(abs(out["eval_csv_step"] - ckpt_step) <= 0.02 * max(ckpt_step, 1))
    ok = True
    for k in ("return", "goals_reached", "dropped", "length"):
        h = summary[k]
        m_csv = float(row[f"{prefix}/{k}"])
        n_h = max(h["n"], 1)
        pooled = (h["mean"] * n_h + m_csv * n_csv) / (n_h + n_csv)
        if k == "return":
            sd_h, sd_csv = h["std"], float(row[f"{prefix}/return_std"])
        elif k == "dropped":
            sd_h = sd_csv = math.sqrt(max(pooled * (1 - pooled), 0.0))
        elif k == "goals_reached":
            sd_h = sd_csv = max(h["std"], math.sqrt(max(pooled, 0.0)))
        else:
            out["metrics"][k] = {"harness": h["mean"], "eval_csv": m_csv, "tested": False}
            continue
        se = math.sqrt(sd_h ** 2 / n_h + sd_csv ** 2 / n_csv)
        z = abs(h["mean"] - m_csv) / se if se > 0 else (0.0 if h["mean"] == m_csv else float("inf"))
        out["metrics"][k] = {"harness": h["mean"], "eval_csv": m_csv, "z": z, "ok": bool(z <= z_tol)}
        ok &= z <= z_tol
    out["ok"] = bool(ok)
    out["z_tol"] = z_tol
    if out.get("comparable") is False:
        out["ok"] = None  # not a valid check: eval row and checkpoint are different agents
    return out
