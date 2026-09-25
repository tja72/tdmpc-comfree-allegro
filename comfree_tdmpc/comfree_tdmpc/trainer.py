"""The closed loop: plan with the simulator, learn value/policy, identify physics.

One iteration of the loop is

    MPPI(sim(theta_hat)) --> transitions --> { TD learning of Q/pi,
                                              SysID of theta_hat }  --> sim(theta_hat)

so the simulator is both what the planner rolls out and what SysID corrects,
and the value function is what gives the short-horizon planner its long-range
information.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.agent.buffer import ReplayBuffer
from comfree_tdmpc.agent.tdmpc_agent import AgentConfig, TDMPCAgent
from comfree_tdmpc.envs.allegro import AllegroReorientEnv, EnvConfig
from comfree_tdmpc.logger import CSVLogger
from comfree_tdmpc.planner.mppi import MPPIPlanner, PlannerConfig
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig
from comfree_tdmpc.sim.batch_sim import load_spec
from comfree_tdmpc.sim.handle_writer import HandleWriter
from comfree_tdmpc.sim.param_space import ParamSpace
from comfree_tdmpc.sysid.stochastic import StochasticSysID, StochasticSysIDConfig


@dataclass
class TrainConfig:
    seed: int = 0
    total_episodes: int = 40
    seed_episodes: int = 2
    updates_per_step: float = 0.5
    buffer_capacity: int = 30000
    # Fraction of every training batch drawn from harvested MPPI elite rollouts
    # rather than from executed environment transitions.
    sim_data_ratio: float = 0.5
    # Env steps before the planner is allowed to use the terminal value at all.
    # An untrained Q is worse than no Q: the planner maximises straight into its
    # errors, which measurably destroys performance at every horizon.
    terminal_value_warmup_steps: int = 1500
    sim_buffer_capacity: int = 60000
    eval_every: int = 5
    eval_episodes: int = 1
    # The policy costs nothing to evaluate (no planning), so average over more
    # episodes: it is the noisiest quantity we track.
    eval_episodes_pi: int = 3
    video_every: int = 10
    save_every: int = 20

    # SysID schedule
    sysid_enabled: bool = True
    sysid_every_steps: int = 300
    sysid_iters: int = 12
    sysid_warmup_steps: int = 300

    # How far "reality" sits from the compiled model, in normalised u units.
    # The planner starts from the model as compiled -- the real-robot situation,
    # where the CAD/URDF numbers are the prior and reality has to be identified.
    reality_perturb: float = 0.25
    reality_seed: int = 12345

    out_dir: str = "outputs/run"
    tag: str = ""
    device: str = "cuda:0"

    env: EnvConfig = field(default_factory=EnvConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    sysid: StochasticSysIDConfig = field(default_factory=StochasticSysIDConfig)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.tag = cfg.tag or self.out.name

        self.space = ParamSpace.realistic(load_spec())
        self.u_model = self.space.true_u.copy()
        self.env = AllegroReorientEnv(
            cfg.env, None, cfg.device, cfg.seed, space=self.space, u=self.u_model
        )
        rng = np.random.default_rng(cfg.reality_seed)
        self.u_reality = self.space.sample_stable(
            rng, cfg.reality_perturb, self._probe_stable
        )
        self.env.writer.write_u(self.u_reality)
        self.device = self.env.torch_device

        self.plan_sim = BatchSim(
            SimConfig(
                nworld=cfg.planner.num_samples,
                n_substeps=cfg.env.n_substeps,
                timestep=cfg.env.timestep,
                device=cfg.device,
            )
        )
        self.plan_writer = HandleWriter(self.plan_sim, self.space)
        self.plan_writer.write_u(self.u_model)

        self.agent = TDMPCAgent(cfg.agent, self.env.obs_dim, self.env.action_dim, self.device)
        self.sim_buffer = None
        if cfg.planner.record_elites:
            self.sim_buffer = ReplayBuffer(
                cfg.sim_buffer_capacity, self.env.obs_dim, self.env.action_dim, 1, 1, self.device
            )
        self.planner = MPPIPlanner(
            self.plan_sim, self.env.task, cfg.planner, agent=self.agent,
            sim_buffer=self.sim_buffer,
        )
        self.buffer = ReplayBuffer(
            cfg.buffer_capacity,
            self.env.obs_dim,
            self.env.action_dim,
            self.env.sim.nq,
            self.env.sim.nv,
            self.device,
        )

        self.sysid = None
        if cfg.sysid_enabled:
            self.sysid = StochasticSysID(
                cfg.sysid, self.space, device=cfg.device,
                n_substeps=cfg.env.n_substeps, timestep=cfg.env.timestep,
            )
            self.sysid.writer.write_u(self.u_reality)  # warm the kernels

        self.train_log = CSVLogger(self.out / "train.csv")
        self.eval_log = CSVLogger(self.out / "eval.csv")
        self.sysid_log = CSVLogger(self.out / "sysid.csv")
        self._use_terminal_value = cfg.planner.use_terminal_value
        self.step = 0
        self.episode = 0
        self.last_sysid_step = 0
        self.renderer = None
        self._start = time.time()

    def _probe_stable(self, u: np.ndarray) -> bool:
        """Does this parameter vector simulate quietly from the scene keyframe?"""
        sim = self.env.sim
        self.env.writer.write_u(u)
        sim.reset_to_key()
        hold = torch.as_tensor(sim.key_ctrl, dtype=torch.float32, device=sim.torch_device)
        for _ in range(40):
            sim.set_ctrl_raw(hold.unsqueeze(0).expand(sim.nworld, -1))
            sim.step()
        sim.sync()
        finite = bool(torch.isfinite(sim.qpos).all() and torch.isfinite(sim.qvel).all())
        settled = finite and float(sim.qvel.abs().max()) < 50.0
        cube_held = settled and float(sim.qpos[0, 18]) > self.cfg.env.task.drop_height
        return bool(cube_held)

    def collect_episode(self, seed_episode: bool = False) -> dict:
        env = self.env
        obs = env.reset()
        self.planner.reset()
        ep_reward, ep_len, updates = 0.0, 0, 0
        angles = []
        t0 = time.perf_counter()
        base_terminal = self._use_terminal_value
        self.planner.cfg.use_terminal_value = (
            base_terminal and self.step >= self.cfg.terminal_value_warmup_steps
        )
        while True:
            qpos, qvel = env.state()
            if seed_episode:
                # Smoothed noise keeps the cube in the hand long enough to be useful.
                a = (0.75 * env.prev_action.squeeze(0) + 0.4 * torch.randn(
                    env.action_dim, device=self.device)).clamp(-1, 1)
                mu, std = a, torch.full_like(a, self.cfg.planner.max_std)
            else:
                a, mu, std = self.planner.plan(
                    qpos, qvel, env.goal_quat, env.prev_action, t0=(ep_len == 0)
                )
            next_obs, reward, done, info = env.step(a)
            self.buffer.add(
                obs, next_obs, a, mu, std, reward,
                float(info["dropped"]), qpos.squeeze(0), qvel.squeeze(0),
            )
            obs = next_obs
            ep_reward += reward
            ep_len += 1
            self.step += 1
            angles.append(info["angle"])

            if not seed_episode and len(self.buffer) >= self.cfg.agent.batch_size // 2:
                n_updates = int(self.cfg.updates_per_step)
                if np.random.rand() < self.cfg.updates_per_step - n_updates:
                    n_updates += 1
                for _ in range(n_updates):
                    metrics = self.agent.update(self.sample_batch())
                    updates += 1
            if done:
                break
        self.buffer.end_episode()
        return {
            "episode_reward": ep_reward,
            "episode_length": ep_len,
            "goals_reached": info["goals_reached"],
            "dropped": info["dropped"],
            "mean_angle": float(np.mean(angles)),
            "min_angle": float(np.min(angles)),
            "updates": updates,
            "sec": time.perf_counter() - t0,
            **({} if seed_episode else self.planner.last_stats),
            **({} if seed_episode or not updates else metrics),
        }

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Mix executed transitions with harvested planner rollouts."""
        bs = self.cfg.agent.batch_size
        ratio = self.cfg.sim_data_ratio
        if self.sim_buffer is None or len(self.sim_buffer) < bs or ratio <= 0:
            return self.buffer.sample(bs)
        n_sim = int(bs * ratio)
        real = self.buffer.sample(bs - n_sim)
        sim = self.sim_buffer.sample(n_sim)
        return {k: torch.cat([real[k], sim[k]], dim=0) for k in real}

    def run_sysid(self) -> dict:
        if self.sysid is None:
            return {}
        cfg = self.cfg

        def sampler():
            return self.buffer.sample_chunks(cfg.sysid.num_chunks, cfg.sysid.chunk_length)

        if sampler() is None:
            return {}
        eval_chunks = sampler()
        # Loss at the true parameters is the floor: what is left above it is model
        # error still on the table.  In this many dimensions the parameters themselves
        # only partly identify, so this is the honest measure of progress.
        ref = float(self.sysid._evaluate(
            np.repeat(self.u_reality[None], self.sysid.n_probe, 0), eval_chunks)[0])
        before = float(self.sysid._evaluate(
            np.repeat(self.u_model[None], self.sysid.n_probe, 0), eval_chunks)[0])

        t0 = time.perf_counter()
        self.u_model, history = self.sysid.optimize(
            self.u_model, sampler, num_iters=cfg.sysid_iters,
            seed=self.step, eval_chunks=eval_chunks,
        )
        self.plan_writer.write_u(self.u_model)
        self.last_sysid_step = self.step

        err = np.abs(self.u_model - self.u_reality)
        after = history[-1].get("val_loss", float("nan")) if history else float("nan")
        row = {
            "step": self.step, "episode": self.episode,
            "sec": time.perf_counter() - t0,
            "loss": history[-1]["loss"] if history else float("nan"),
            "val_loss": after, "ref_loss": ref, "loss_before": before,
            "gap_closed": float(100 * (before - after) / max(before - ref, 1e-12)),
            "param_err_mean": float(err.mean()),
        }
        for g, idx in self.space.groups().items():
            row[f"err/{g}"] = float(err[idx].mean())
        self.sysid_log.log(row)
        return row

    @torch.no_grad()
    def evaluate(self, num_episodes: int, use_pi: bool = False, record: bool = False) -> dict:
        """Roll out either the planner or the distilled policy alone."""
        env = self.env
        returns, solved, drops, min_angles = [], [], [], []
        frames = []
        for i in range(num_episodes):
            obs = env.reset()
            self.planner.reset()
            ep_r, angles, t = 0.0, [], 0
            while True:
                if use_pi:
                    a = self.agent.act(obs, eval_mode=True)
                else:
                    qpos, qvel = env.state()
                    a, _, _ = self.planner.plan(
                        qpos, qvel, env.goal_quat, env.prev_action, t0=(t == 0), eval_mode=True
                    )
                obs, r, done, info = env.step(a)
                ep_r += r
                angles.append(info["angle"])
                if record and i == 0:
                    frames.append(self._frame(info, use_pi))
                t += 1
                if done:
                    break
            returns.append(ep_r)
            solved.append(info["goals_reached"])
            drops.append(info["dropped"])
            min_angles.append(float(np.min(angles)))
        out = {
            "return": float(np.mean(returns)),
            "goals_reached": float(np.mean(solved)),
            "dropped": float(np.mean(drops)),
            "min_angle": float(np.mean(min_angles)),
        }
        if record:
            out["frames"] = frames
        return out

    def _frame(self, info: dict, use_pi: bool = False) -> np.ndarray:
        if self.renderer is None:
            from comfree_tdmpc.render import SceneRenderer

            self.renderer = SceneRenderer()
        cfg = self.cfg
        who = (
            "LEARNED POLICY only (no planning)"
            if use_pi
            else f"MPPI planner  H={cfg.planner.horizon}  {cfg.planner.num_samples} samples"
        )
        theta_err = float(np.abs(self.u_model - self.u_reality).mean())
        hold = info.get("hold_steps", 0)
        streak = info.get("success_streak", 0)
        status = (
            f"AT GOAL  holding {streak}/{hold}"
            if info.get("success", 0) > 0
            else f"goal err={info['angle']:.2f} rad"
        )
        return self.renderer.frame(
            self.env.qpos[0].cpu().numpy(),
            self.env.goal_quat[0].cpu().numpy(),
            label=f"{self.tag}   {who}\n"
            f"train episode {self.episode}   {status}\n"
            f"goals solved={info['goals_reached']}   model error={theta_err:.2f}",
            at_goal=info.get("success", 0) > 0,
        )

    def train(self) -> None:
        cfg = self.cfg
        print(f"[trainer] out={self.out}  device={self.device}")
        print(f"[trainer] {self.space.describe()}")
        print(f"[trainer] reality is {np.abs(self.u_reality - self.u_model).mean():.4f} away "
              f"from the compiled model, in normalised units")

        for ep in range(cfg.total_episodes):
            self.episode = ep
            seed_ep = ep < cfg.seed_episodes
            stats = self.collect_episode(seed_episode=seed_ep)
            row = {
                "episode": ep,
                "step": self.step,
                "wall_time": time.time() - self._start,
                "seed_episode": float(seed_ep),
                "terminal_value_on": float(self.planner.cfg.use_terminal_value),
                "sim_buffer": float(len(self.sim_buffer)) if self.sim_buffer else 0.0,
                "param_err_mean": float(np.abs(self.u_model - self.u_reality).mean()),
                **stats,
            }
            self.train_log.log(row)
            print(
                f"[ep {ep:3d}] step={self.step:5d} R={stats['episode_reward']:8.1f} "
                f"len={stats['episode_length']:3d} solved={stats['goals_reached']} "
                f"drop={stats['dropped']:.0f} theta_err={row['param_err_mean']:.3f} "
                f"({stats['sec']:.0f}s)",
                flush=True,
            )

            if (
                cfg.sysid_enabled
                and self.step - self.last_sysid_step >= cfg.sysid_every_steps
                and self.step >= cfg.sysid_warmup_steps
            ):
                s = self.run_sysid()
                if s:
                    print(
                        f"      [sysid] model err {s['param_err_mean']:.4f}  loss "
                        f"{s['loss_before']:.4g} -> {s['val_loss']:.4g} "
                        f"(floor {s['ref_loss']:.4g}, {s['gap_closed']:.0f}% closed) "
                        f"[{s['sec']:.1f}s]",
                        flush=True,
                    )

            if (ep + 1) % cfg.eval_every == 0 or ep == cfg.total_episodes - 1:
                record = (ep + 1) % cfg.video_every == 0 or ep == cfg.total_episodes - 1
                ev = self.evaluate(cfg.eval_episodes, record=record)
                frames = ev.pop("frames", None)
                ev_pi = self.evaluate(cfg.eval_episodes_pi, use_pi=True, record=record)
                frames_pi = ev_pi.pop("frames", None)
                erow = {
                    "episode": ep,
                    "step": self.step,
                    "param_err_mean": row["param_err_mean"],
                    **{f"mpc/{k}": v for k, v in ev.items()},
                    **{f"pi/{k}": v for k, v in ev_pi.items()},
                }
                self.eval_log.log(erow)
                print(
                    f"      [eval] mpc R={ev['return']:.1f} solved={ev['goals_reached']:.1f} "
                    f"| pi R={ev_pi['return']:.1f} solved={ev_pi['goals_reached']:.1f}",
                    flush=True,
                )
                if frames:
                    from comfree_tdmpc.render import save_gif

                    save_gif(frames, self.out / "videos" / f"ep{ep:04d}_mpc.gif", fps=25)
                if frames_pi:
                    from comfree_tdmpc.render import save_gif

                    save_gif(frames_pi, self.out / "videos" / f"ep{ep:04d}_pi.gif", fps=25)

            if (ep + 1) % cfg.save_every == 0 or ep == cfg.total_episodes - 1:
                self.agent.save(str(self.out / "agent.pt"))
                np.save(self.out / "params.npy", self.u_model)
                np.save(self.out / "params_reality.npy", self.u_reality)

        self.train_log.save_json(self.out / "config.json", asdict(cfg))
        print(f"[trainer] done in {(time.time()-self._start)/60:.1f} min")
