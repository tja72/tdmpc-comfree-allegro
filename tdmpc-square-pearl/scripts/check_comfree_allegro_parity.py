"""
Parity checks for task=comfree-allegro-cube (tdmpc_square/envs/comfree_allegro.py).

The TD-MPC2 baseline is only a fair comparison against ../comfree_tdmpc if the
wrapper is the very same task in the very same reality. Run from the repo root
before any long run:

    MUJOCO_GL=egl python scripts/check_comfree_allegro_parity.py

(a) wrapper vs. VecAllegroReorientEnv directly: from the same state + action, same
    obs/reward up to the sim's own step-to-step noise floor (the GPU sim is not
    bitwise deterministic; measured alongside), identical done
    flags, goal counts and seeded resets, across several episode boundaries
(b) wrapper reality u == the refactored helper == a verbatim copy of the old
    inline VecTrainer code (16 lanes) == outputs/ablation/d_termq/params_reality.npy
(c) 70-dim obs, shapes, dtypes, bounds, max_episode_steps == 120, via make_env + TensorWrapper
(d) a drop terminates the episode (terminated, not truncated) with the -w_drop penalty
(e) the saved d_termq policy, run through the wrapper, solves goals in the range of
    its logged pi eval
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from comfree_tdmpc.agent.tdmpc_agent import AgentConfig, TDMPCAgent
from comfree_tdmpc.envs.vec_allegro import VecAllegroReorientEnv, VecEnvConfig
from comfree_tdmpc.reality import apply_reality
from comfree_tdmpc.sim.batch_sim import load_spec
from comfree_tdmpc.sim.param_space import ParamSpace
import comfree_tdmpc

from tdmpc_square.envs import make_env
from tdmpc_square.envs.comfree_allegro import ComFreeAllegroEnv

COMFREE_ROOT = Path(comfree_tdmpc.__file__).resolve().parents[1]
D_TERMQ = COMFREE_ROOT / "outputs" / "ablation" / "d_termq"
SEED = 0
FAILED = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def old_inline_reality(env, space, reality_seed=12345, reality_perturb=0.25):
    """Verbatim copy of VecTrainer.__init__ + _probe_stable before the refactor."""

    def _probe_stable(u):
        sim = env.sim
        env.writer.write_u(u)
        sim.reset_to_key()
        hold = torch.as_tensor(sim.key_ctrl, dtype=torch.float32, device=sim.torch_device)
        for _ in range(40):
            sim.set_ctrl_raw(hold.unsqueeze(0).expand(sim.nworld, -1))
            sim.step()
        sim.sync()
        finite = bool(torch.isfinite(sim.qpos).all() and torch.isfinite(sim.qvel).all())
        settled = finite and float(sim.qvel.abs().max()) < 50.0
        return bool(settled and float(sim.qpos[0, 18]) > env.cfg.task.drop_height)

    rng = np.random.default_rng(reality_seed)
    u = (
        space.sample_stable(rng, reality_perturb, _probe_stable)
        if reality_perturb > 0
        else space.true_u.copy()
    )
    env.writer.write_u(u)
    return u


def check_a(wrapper):
    space = ParamSpace.realistic(load_spec())
    direct = VecAllegroReorientEnv(
        VecEnvConfig(num_envs=1, episode_length=120), "cuda:0", SEED,
        space=space, u=space.true_u.copy(),
    )
    apply_reality(direct, space)
    # The ComFree GPU sim is not bitwise deterministic (float atomics in the contact
    # solver: replaying one env from a snapshot differs by ~1e-6 after a few steps,
    # and contact chaos amplifies that), so whole trajectories cannot be compared.
    # Instead, before every wrapper step the direct env is restored to the wrapper's
    # exact state (incl. RNG) and stepped with the same action.
    obs_w, _ = wrapper.reset()
    obs_d = direct.reset()[0].cpu().numpy()
    reset_exact = np.array_equal(obs_w, obs_d)
    rng = np.random.default_rng(1)
    max_obs, max_r, n_eps, n_steps, mismatch = 0.0, 0.0, 0, 0, 0
    noise_obs, noise_r = 0.0, 0.0
    while n_eps < 4:
        a = rng.uniform(-1, 1, 16).astype(np.float32)
        snap = wrapper.env.snapshot()
        # Noise floor: the direct env stepped twice from this same state.
        direct.restore(snap)
        n1, r1, _, _ = direct.step(torch.as_tensor(a).unsqueeze(0))
        direct.restore(snap)
        n2, r2, _, _ = direct.step(torch.as_tensor(a).unsqueeze(0))
        noise_obs = max(noise_obs, (n1 - n2).abs().max().item())
        noise_r = max(noise_r, abs(float(r1[0]) - float(r2[0])))
        direct.restore(snap)
        obs_w, r_w, term, trunc, info = wrapper.step(a)
        nobs_d, r_d, done_d, info_d = direct.step(torch.as_tensor(a).unsqueeze(0))
        max_obs = max(max_obs, np.abs(obs_w - nobs_d[0].cpu().numpy()).max())
        max_r = max(max_r, abs(r_w - float(r_d[0])))
        mismatch += int(
            (term or trunc) != bool(done_d[0])
            or term != bool(info_d["dropped"][0])
            or trunc != bool(info_d["truncated"][0])
            or info["goals_reached"] != float(info_d["goals_reached"][0])
        )
        n_steps += 1
        if term or trunc:
            n_eps += 1
            direct.restore(wrapper.env.snapshot())
            obs_w, _ = wrapper.reset()
            obs_d = direct.reset()[0].cpu().numpy()
            reset_exact &= np.array_equal(obs_w, obs_d)
    check(
        "(a) wrapper == direct VecAllegroReorientEnv (per step, from identical state)",
        reset_exact and mismatch == 0 and max_obs < 1e-3 and max_r < 1e-2
        and max_obs <= 10 * max(noise_obs, 1e-6) and max_r <= 10 * max(noise_r, 1e-6),
        f"{n_steps} steps / {n_eps} episodes, resets exact={reset_exact}, "
        f"max|dobs|={max_obs:.1e} (sim noise floor {noise_obs:.1e}), "
        f"max|dr|={max_r:.1e} (floor {noise_r:.1e}), done/goal mismatches={mismatch}",
    )


def check_b(wrapper):
    space = ParamSpace.realistic(load_spec())
    env16 = VecAllegroReorientEnv(
        VecEnvConfig(), "cuda:0", SEED, space=space, u=space.true_u.copy()
    )
    u_old = old_inline_reality(env16, space)
    u_new = apply_reality(env16, space)
    u_run = np.load(D_TERMQ / "params_reality.npy")
    check("(b) helper == old inline VecTrainer code (bit-identical)", np.array_equal(u_new, u_old))
    check("(b) wrapper u == VecTrainer u (16 lanes)", np.array_equal(wrapper.u_reality, u_new))
    check("(b) wrapper u == d_termq/params_reality.npy", np.array_equal(wrapper.u_reality, u_run))
    check("(b) reality is not the nominal model", not np.array_equal(u_new, space.true_u),
          f"mean|u-true|={np.abs(u_new - space.true_u).mean():.3f}")


def check_c():
    cfg = OmegaConf.create(dict(task="comfree-allegro-cube", multitask=False, obs="state",
                                seed=SEED, seed_steps="???"))
    env = make_env(cfg)
    obs, _ = env.reset()
    a = env.rand_act()
    obs2, r, term, trunc, info = env.step(a)
    ok = (
        tuple(env.observation_space.shape) == (70,)
        and tuple(env.action_space.shape) == (16,)
        and np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
        and env.max_episode_steps == 120 and cfg.episode_length == 120
        and cfg.action_dim == 16 and tuple(cfg.obs_shape["state"]) == (70,)
        and cfg.seed_steps == 1000
        and obs.dtype == torch.float32 and tuple(obs.shape) == (70,)
        and obs2.dtype == torch.float32 and r.dtype == torch.float32
        and a.dtype == torch.float32 and a.abs().max() <= 1
        and "success" in info and "success_subtasks" in info
        and "goals_reached" in info and "dropped" in info
    )
    check("(c) shapes/bounds/dtypes via make_env + TensorWrapper", ok,
          f"obs {tuple(obs.shape)} {obs.dtype}, act {tuple(env.action_space.shape)} "
          f"[{env.action_space.low.min()}, {env.action_space.high.max()}], "
          f"max_episode_steps={env.max_episode_steps}, seed_steps={cfg.seed_steps}")
    # a full random episode runs to the time limit or a drop, never beyond
    env.reset()
    for t in range(1, 200):
        _, _, term, trunc, info = env.step(env.rand_act())
        if term or trunc:
            break
    check("(c) random episode ends by step 120", t <= 120 and (trunc == (t == 120) or term),
          f"ended at t={t}, terminated={term}, truncated={trunc}")
    ep = info.get("episode")
    check("(c) terminal step carries comfree's episode record",
          ep is not None and ep["episode_length"] == t
          and ep["goals_reached"] == info["goals_reached"],
          f"{ep}")
    return env


def check_d(wrapper):
    w_drop = wrapper.env.cfg.task.w_drop
    wrapper.reset()
    for _ in range(5):
        wrapper.step(wrapper.env.key_action.cpu().numpy())
    # Teleport the cube well below the drop height; the next step must end the episode.
    sim = wrapper.env.sim
    sim.qpos[0, 18] = wrapper.env.cfg.task.drop_height - 0.2
    sim.forward()
    _, r, term, trunc, info = wrapper.step(wrapper.env.key_action.cpu().numpy())
    check("(d) drop -> terminated, not truncated, penalty applied",
          term and not trunc and info["dropped"] == 1.0 and r <= -w_drop + 10,
          f"terminated={term}, truncated={trunc}, reward={r:.2f}, w_drop={w_drop}")
    # Natural drops under random actions, as a sanity check of the rate.
    rng = np.random.default_rng(2)
    drops, lengths = 0, []
    for _ in range(10):
        wrapper.reset()
        for t in range(1, 121):
            _, _, term, trunc, _ = wrapper.step(rng.uniform(-1, 1, 16).astype(np.float32))
            if term or trunc:
                break
        drops += int(term)
        lengths.append(t)
    print(f"      random actions: {drops}/10 episodes dropped, mean length {np.mean(lengths):.1f}")


def check_e(wrapper, n_episodes=16):
    cfg = json.loads((D_TERMQ / "config.json").read_text())
    agent = TDMPCAgent(AgentConfig(**cfg["agent"]), 70, 16, torch.device("cuda"))
    agent.load(str(D_TERMQ / "agent.pt"))
    agent.eval()
    goals, drops = [], []
    for _ in range(n_episodes):
        obs, _ = wrapper.reset()
        while True:
            with torch.no_grad():
                a = agent.act(torch.as_tensor(obs, device="cuda"), eval_mode=True)
            obs, _, term, trunc, info = wrapper.step(a.cpu().numpy())
            if term or trunc:
                break
        goals.append(info["goals_reached"])
        drops.append(info["dropped"])
    ev = pd.read_csv(D_TERMQ / "eval.csv")
    tail = ev["pi/goals_reached"].iloc[-6:]
    lo, hi = tail.min(), tail.max()
    g = float(np.mean(goals))
    # 16 episodes of a policy that drops half the time: allow some slack around
    # the spread of the last 6 logged evals.
    ok = (lo - 0.35) <= g <= (hi + 0.35)
    check("(e) d_termq pi through wrapper matches its logged pi eval", ok,
          f"wrapper: goals={g:.3f} dropped={np.mean(drops):.3f} over {n_episodes} eps; "
          f"logged last 6 evals goals in [{lo:.3f}, {hi:.3f}], "
          f"dropped in [{ev['pi/dropped'].iloc[-6:].min():.3f}, "
          f"{ev['pi/dropped'].iloc[-6:].max():.3f}]")


def main():
    wrapper = ComFreeAllegroEnv(seed=SEED)
    check_b(wrapper)
    check_a(ComFreeAllegroEnv(seed=SEED))
    check_c()
    check_d(wrapper)
    check_e(wrapper)
    print("ALL PASSED" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
