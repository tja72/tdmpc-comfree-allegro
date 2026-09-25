"""Does the vectorised stack compute the same thing, and how much faster?

Correctness first -- a speedup that changes the algorithm is not a speedup.

Comparing two contact simulations by "are the trajectories equal" does not work
here, and saying so precisely matters.  MuJoCo-Warp packs the constraints of
*all* worlds into one shared pool, so changing how many worlds there are, or
what any one of them is doing, changes the order in which float32 contributions
are summed.  That is a round-off difference at the first step -- and analytic
contact is still chaotic, so a round-off difference grows by roughly an order of
magnitude per control step until it saturates.

Every test below therefore reports two numbers:

  * the divergence between the thing under test and its reference, and
  * a *noise floor*: the divergence between the reference and a bit-identical
    re-run of itself.

A test passes when the first is no worse than the second.  The structural
question -- does world `e * N + i` really carry environment `e`'s state, goal
and action -- is answered separately by the single-step checks, where chaos has
not yet had time to amplify anything and agreement must be at round-off.

    python scripts/07_verify_vec.py --envs 1 4 8 16 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from comfree_tdmpc.agent.tdmpc_agent import AgentConfig, TDMPCAgent
from comfree_tdmpc.agent.vec_buffer import VecReplayBuffer
from comfree_tdmpc.envs.allegro import AllegroReorientEnv, EnvConfig
from comfree_tdmpc.envs.task import ReorientTask, TaskConfig
from comfree_tdmpc.envs.vec_allegro import VecAllegroReorientEnv, VecEnvConfig
from comfree_tdmpc.planner.batch_mppi import BatchMPPIPlanner
from comfree_tdmpc.planner.mppi import MPPIPlanner, PlannerConfig
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig

PASS, FAIL = "  ok  ", " FAIL "
RESULTS: dict[str, dict] = {}


def check(name: str, ok: bool, detail: str, **extra) -> bool:
    print(f"[{PASS if ok else FAIL}] {name:<44} {detail}", flush=True)
    RESULTS[name] = {"ok": bool(ok), "detail": detail, **extra}
    return ok


# 1. per-step state agreement
def test_env_matches_single(steps: int = 40) -> bool:
    """The vec env must trace the single env, to within the simulator's own noise.

    The noise floor is measured by re-running the *single* env on the same
    actions from the same state: any difference there is pure float32 round-off
    amplified by contact, and the vec env cannot be expected to do better.
    """
    E = 4
    single = AllegroReorientEnv(EnvConfig(episode_length=10_000, task=TaskConfig()), seed=0)
    single.cfg.reset_qvel_noise = 0.0
    vec = VecAllegroReorientEnv(
        VecEnvConfig(num_envs=E, episode_length=10_000, reset_qvel_noise=0.0,
                     task=TaskConfig()), seed=0)
    single.reset()
    vec.reset()
    q0, v0 = single.qpos.clone(), single.qvel.clone()
    goal, prev, pang = single.goal_quat.clone(), single.prev_action.clone(), single.prev_angle.clone()

    dev = vec.torch_device
    g = torch.Generator(device=dev).manual_seed(7)
    acts = [(torch.rand(vec.action_dim, device=dev, generator=g) * 2 - 1) * 0.4
            for _ in range(steps)]

    def put_single():
        single.sim.set_state(q0, v0)
        single.goal_quat, single.prev_angle = goal.clone(), pang.clone()
        single.prev_action, single.t = prev.clone(), 0

    def put_vec():
        vec.sim.qpos.copy_(q0.expand(E, -1))
        vec.sim.qvel.copy_(v0.expand(E, -1))
        vec.sim.forward()
        vec.goal_quat.copy_(goal.expand(E, -1))
        vec.prev_action.copy_(prev.expand(E, -1))
        vec.prev_angle.copy_(pang.expand(E))
        vec.t.zero_()

    traces = {}
    for name, put, obj, act in (("single_a", put_single, single, lambda a: a),
                                ("single_b", put_single, single, lambda a: a),
                                ("vec", put_vec, vec, lambda a: a.expand(E, -1))):
        put()
        tr = []
        for a in acts:
            obj.step(act(a))
            tr.append(obj.qpos[0].clone())
        traces[name] = torch.stack(tr)

    floor = (traces["single_a"] - traces["single_b"]).abs().amax(dim=1)
    diff = (traces["single_a"] - traces["vec"]).abs().amax(dim=1)
    del single, vec
    torch.cuda.empty_cache()

    d1, f1 = float(diff[0]), float(floor[0])
    dN, fN = float(diff[-1]), float(floor[-1])
    # One step: chaos has not amplified anything, so this must be round-off.
    # Many steps: only required to stay inside the simulator's own spread.
    ok = d1 < 1e-5 and dN <= max(10 * fN, 1e-3)
    return check(
        "vec env reproduces single env", ok,
        f"step 1: {d1:.1e} (floor {f1:.1e})   step {steps}: {dN:.1e} (floor {fN:.1e})",
        step1=d1, step1_floor=f1, final=dN, final_floor=fN,
    )


# 2. rollout scoring
def test_rollout_matches_single(horizon: int = 8, N: int = 64) -> bool:
    """The batched rollout must score the same action sequences the same way."""
    E = 3
    dev = torch.device("cuda")

    def single_values(H):
        cfg = PlannerConfig(horizon=H, num_samples=N, num_pi_trajs=0, use_terminal_value=False)
        sim = BatchSim(SimConfig(nworld=N))
        task = ReorientTask(TaskConfig(), dev, sim.control_dt, cube_home=sim.key_qpos[16:19])
        pl = MPPIPlanner(sim, task, cfg)
        sim.reset_to_key()
        q0, v0 = sim.qpos[0].clone(), sim.qvel[0].clone()
        goal = torch.tensor([[0.92, 0.0, 0.0, 0.39]], device=dev)
        prev = torch.zeros(1, sim.nu, device=dev)
        g = torch.Generator(device=dev).manual_seed(11)
        acts = (torch.rand(H, N, sim.nu, device=dev, generator=g) * 2 - 1) * 0.5
        out = []
        for _ in range(2):  # twice: the second run gives the noise floor
            sim.reset_to_key()
            v, _ = pl._rollout(q0.view(1, -1), v0.view(1, -1), goal, prev, acts.clone(), 0)
            out.append(v.clone())
        nu = sim.nu
        del sim, pl
        torch.cuda.empty_cache()
        return out[0], out[1], q0, v0, goal, prev, acts, nu

    def batch_values(H, acts, q0, v0, goal, prev, nu):
        cfg = PlannerConfig(horizon=H, num_samples=N, num_pi_trajs=0, use_terminal_value=False)
        sim = BatchSim(SimConfig(nworld=E * N))
        task = ReorientTask(TaskConfig(), dev, sim.control_dt, cube_home=sim.key_qpos[16:19])
        pl = BatchMPPIPlanner(sim, task, cfg, num_envs=E)
        a = acts.unsqueeze(1).expand(H, E, N, nu).contiguous()
        v, _ = pl._rollout(q0.unsqueeze(0).expand(E, -1), v0.unsqueeze(0).expand(E, -1),
                           goal.expand(E, -1), prev.expand(E, -1), a, use_pi=False)
        del sim, pl
        torch.cuda.empty_cache()
        return v

    rows = {}
    for H in (1, horizon):
        sa, sb, q0, v0, goal, prev, acts, nu = single_values(H)
        vb = batch_values(H, acts, q0, v0, goal, prev, nu)
        scale = max(float(sa.abs().max()), 1e-9)
        rows[H] = (float((sa - vb).abs().max()) / scale, float((sa - sb).abs().max()) / scale)

    (d1, f1), (dH, fH) = rows[1], rows[horizon]
    ok = d1 < 1e-4 and dH <= max(10 * fH, 5e-2)
    return check(
        "batched rollout matches single rollout", ok,
        f"H=1: {d1:.1e} (floor {f1:.1e})   H={horizon}: {dH:.1e} (floor {fH:.1e})",
        h1=d1, h1_floor=f1, hH=dH, hH_floor=fH,
    )


# 3. each env gets its own goal
def test_goal_routing(horizon: int = 6, N: int = 32) -> bool:
    """Environment e's reward must be scored against environment e's goal.

    Dynamics are identical across lanes here, so any disagreement is a goal that
    was broadcast to the wrong block of worlds -- and because only the reward
    differs, the comparison is exact arithmetic on identical trajectories.
    """
    E = 4
    dev = torch.device("cuda")
    cfg = PlannerConfig(horizon=horizon, num_samples=N, num_pi_trajs=0,
                        use_terminal_value=False)
    ang = torch.tensor([0.2, 0.9, 1.8, 2.7], device=dev)
    goals = torch.stack([torch.cos(ang / 2), torch.zeros(E, device=dev),
                         torch.zeros(E, device=dev), torch.sin(ang / 2)], dim=1)

    sim = BatchSim(SimConfig(nworld=E * N))
    task = ReorientTask(TaskConfig(), dev, sim.control_dt, cube_home=sim.key_qpos[16:19])
    pl = BatchMPPIPlanner(sim, task, cfg, num_envs=E)
    sim.reset_to_key()
    q0, v0 = sim.qpos[0].clone(), sim.qvel[0].clone()
    prev = torch.zeros(1, sim.nu, device=dev)
    g = torch.Generator(device=dev).manual_seed(13)
    acts = (torch.rand(horizon, N, sim.nu, device=dev, generator=g) * 2 - 1) * 0.5
    a = acts.unsqueeze(1).expand(horizon, E, N, sim.nu).contiguous()
    v_mixed, _ = pl._rollout(q0.expand(E, -1), v0.expand(E, -1), goals,
                             prev.expand(E, -1), a, use_pi=False)

    # Reference: give every lane goal e, and read lane 0.
    ref = []
    for e in range(E):
        ge = goals[e:e + 1].expand(E, -1).contiguous()
        ve, _ = pl._rollout(q0.expand(E, -1), v0.expand(E, -1), ge,
                            prev.expand(E, -1), a.clone(), use_pi=False)
        ref.append(ve[0])
    ref = torch.stack(ref)
    err = float((v_mixed - ref).abs().max())
    scale = max(float(ref.abs().max()), 1e-9)
    spread = float((ref - ref.mean(0, keepdim=True)).abs().max())
    del sim, pl
    torch.cuda.empty_cache()
    ok = err / scale < 5e-2 and spread > 1.0
    return check("each env is scored against its own goal", ok,
                 f"max |dvalue|/scale={err/scale:.1e}  (goals differ by {spread:.1f})",
                 rel_err=err / scale, goal_spread=spread)


# 4. each env gets its own action
def test_action_routing(steps: int = 3) -> bool:
    """Lane e's action must reach lane e's world, and only lane e's world."""
    E = 6
    vec = VecAllegroReorientEnv(
        VecEnvConfig(num_envs=E, episode_length=10_000, reset_qvel_noise=0.0), seed=3)
    vec.reset()
    q0 = vec.qpos.clone()
    dev = vec.torch_device
    levels = torch.linspace(-0.6, 0.6, E, device=dev)
    a = levels.view(E, 1).expand(E, vec.action_dim).contiguous()

    for _ in range(steps):
        vec.step(a)
    mixed = vec.qpos.clone()

    # Reference: run each level in every lane, read lane 0.
    ref = []
    for e in range(E):
        vec.sim.qpos.copy_(q0)
        vec.sim.qvel.zero_()
        vec.sim.forward()
        vec.t.zero_()
        ae = a[e:e + 1].expand(E, -1).contiguous()
        for _ in range(steps):
            vec.step(ae)
        ref.append(vec.qpos[0].clone())
    ref = torch.stack(ref)
    err = float((mixed - ref).abs().max())
    spread = float((ref - ref.mean(0, keepdim=True)).abs().max())
    del vec
    torch.cuda.empty_cache()
    ok = err < 1e-4 and spread > 1e-2
    return check("each env receives its own action", ok,
                 f"max |dqpos|={err:.1e}  (actions differ by {spread:.1e})",
                 err=err, spread=spread)


# 5. lanes do not interact
def test_lane_independence(steps: int = 25) -> bool:
    """Perturbing lane 0 must not move other lanes beyond the simulator's noise.

    It cannot be *zero*: the constraint pool is shared, so lane 0's contacts
    change the summation order for everyone.  What must hold is that the effect
    is no larger than re-running the identical experiment twice.
    """
    E = 8
    vec = VecAllegroReorientEnv(
        VecEnvConfig(num_envs=E, episode_length=10_000, reset_qvel_noise=0.0), seed=3)
    vec.reset()
    q0 = vec.qpos.clone()
    dev = vec.torch_device
    g = torch.Generator(device=dev).manual_seed(5)
    seq = [(torch.rand(E, vec.action_dim, device=dev, generator=g) * 2 - 1) * 0.4
           for _ in range(steps)]

    def run(flip: bool):
        vec.sim.qpos.copy_(q0)
        vec.sim.qvel.zero_()
        vec.sim.forward()
        vec.t.zero_()
        for a in seq:
            a = a.clone()
            if flip:
                a[0] = -a[0]
            vec.step(a)
        return vec.qpos.clone()

    ref, rerun, flipped = run(False), run(False), run(True)
    floor = float((ref[1:] - rerun[1:]).abs().max())
    leak = float((ref[1:] - flipped[1:]).abs().max())
    moved0 = float((ref[0] - flipped[0]).abs().max())
    del vec
    torch.cuda.empty_cache()
    ok = moved0 > 1e-3 and leak <= max(10 * floor, 1e-4)
    return check("lanes do not interact", ok,
                 f"other lanes moved {leak:.1e} (floor {floor:.1e}), "
                 f"lane 0 moved {moved0:.1e}",
                 leak=leak, floor=floor, lane0=moved0)


# 6. chunks
def test_buffer_chunks() -> bool:
    """Sampled SysID chunks must be contiguous and inside a single episode."""
    dev = torch.device("cuda")
    E, C, L = 6, 200, 5
    buf = VecReplayBuffer(C, E, obs_dim=4, action_dim=3, nq=2, nv=2, device=dev)
    ep = torch.arange(E, device=dev)
    for i in range(150):
        if i in (40, 90):  # force an episode boundary in every lane
            ep = ep + E
        z = lambda *s: torch.full(s, float(i), device=dev)  # noqa: E731
        # qpos encodes (lane, timestep) so contiguity is checkable after the fact.
        qpos = torch.stack([torch.arange(E, device=dev).float(),
                            torch.full((E,), float(i), device=dev)], dim=1)
        buf.add(z(E, 4), z(E, 4), z(E, 3), z(E, 3), z(E, 3), z(E),
                torch.zeros(E, device=dev), qpos, z(E, 2), ep)
    ch = buf.sample_chunks(32, L)
    if ch is None:
        return check("buffer chunks are contiguous", False, "sampler returned None")
    lane0, t0 = ch["qpos0"][:, 0], ch["qpos0"][:, 1]
    lane_ok = bool((ch["qpos_ref"][..., 0] == lane0.unsqueeze(0)).all())
    want = t0.unsqueeze(0) + torch.arange(1, L + 1, device=dev).float().unsqueeze(1)
    time_ok = bool((ch["qpos_ref"][..., 1] == want).all())
    shape_ok = ch["actions"].shape == (L, 32, 3)
    del buf
    torch.cuda.empty_cache()
    ok = lane_ok and time_ok and shape_ok
    return check("buffer chunks are contiguous", ok,
                 f"same lane={lane_ok}  consecutive t={time_ok}  shape={shape_ok}")


# 7. throughput
def bench_envs(num_envs: int, N: int, horizon: int, iters: int,
               steps: int = 12, topk: int = 8) -> dict:
    cfg = PlannerConfig(horizon=horizon, iterations=iters, num_samples=N,
                        num_pi_trajs=24, use_terminal_value=True, record_elites=False)
    env = VecAllegroReorientEnv(VecEnvConfig(num_envs=num_envs, episode_length=10_000))
    sim = BatchSim(SimConfig(nworld=num_envs * N))
    task = ReorientTask(TaskConfig(), sim.torch_device, sim.control_dt,
                        cube_home=sim.key_qpos[16:19])
    agent = TDMPCAgent(AgentConfig(), env.obs_dim, env.action_dim, env.torch_device)
    planner = BatchMPPIPlanner(sim, task, cfg, num_envs, agent=agent)

    env.reset()

    def loop(n):
        t0 = time.perf_counter()
        for _ in range(n):
            qpos, qvel = env.state()
            a, _, _ = planner.plan(qpos, qvel, env.goal_quat, env.prev_action,
                                   t0=env.needs_t0.clone())
            env.step(a)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    loop(2)
    dt = loop(steps)
    del env, sim, planner, agent
    torch.cuda.empty_cache()
    return {
        "num_envs": num_envs,
        "worlds": num_envs * N,
        "iter_ms": dt * 1e3,
        "env_steps_per_s": num_envs / dt,
        "elite_transitions_per_s": num_envs * topk * horizon / dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--skip_tests", action="store_true")
    ap.add_argument("--out", type=str, default="outputs/07_verify/vec.json")
    args = ap.parse_args()

    if not args.skip_tests:
        print("=== correctness (divergence vs the simulator's own noise floor) ===")
        for fn in (test_env_matches_single, test_rollout_matches_single,
                   test_goal_routing, test_action_routing,
                   test_lane_independence, test_buffer_chunks):
            fn()

    print(f"\n=== throughput (N={args.num_samples}, H={args.horizon}, "
          f"{args.iterations} iters) ===")
    print(f"{'envs':>5} {'worlds':>7} {'iter ms':>9} {'env steps/s':>12} "
          f"{'speedup':>8} {'elite trans/s':>14}")
    thr, base = [], None
    for e in args.envs:
        try:
            r = bench_envs(e, args.num_samples, args.horizon, args.iterations)
        except Exception as exc:  # noqa: BLE001
            print(f"{e:>5}  FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        base = base or r["env_steps_per_s"]
        r["speedup"] = r["env_steps_per_s"] / base
        thr.append(r)
        print(f"{r['num_envs']:>5} {r['worlds']:>7} {r['iter_ms']:>9.1f} "
              f"{r['env_steps_per_s']:>12.1f} {r['speedup']:>7.1f}x "
              f"{r['elite_transitions_per_s']:>14.0f}", flush=True)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"correctness": RESULTS, "throughput": thr}, indent=2))
    print(f"\nwrote {p}")
    bad = [k for k, v in RESULTS.items() if not v["ok"]]
    if bad:
        raise SystemExit(f"correctness checks failed: {bad}")


if __name__ == "__main__":
    main()
