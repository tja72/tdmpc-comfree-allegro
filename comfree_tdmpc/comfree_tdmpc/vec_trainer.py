"""The closed loop, run on `num_envs` environments at once.

Same algorithm as `trainer.py` -- plan with the simulator, learn Q and pi from
what the planner did, identify the simulator's parameters from what actually
happened -- but organised around a step budget rather than an episode count,
because vectorised lanes terminate at different times.

The three things this loop can switch off are exactly the three claims of the
project, which is what the ablation measures:

  distill      harvest the MPPI elite rollouts as extra training data
  sysid        correct the planner's physical parameters from replayed chunks
  terminal Q   let the planner add gamma^H Q(s_H) to every sampled rollout

`pure_rl` is a fourth, exclusive mode: no planner is built at all, the policy
acts in the real env directly (TD-MPC2's plain SAC-style actor-critic), and
the three switches above -- which all exist to make planner-collected data
better -- are asserted off rather than silently ignored.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.agent.buffer import ReplayBuffer
from comfree_tdmpc.agent.tdmpc_agent import AgentConfig, TDMPCAgent
from comfree_tdmpc.agent.vec_buffer import VecReplayBuffer
from comfree_tdmpc.envs.vec_allegro import VecAllegroReorientEnv, VecEnvConfig
from comfree_tdmpc.logger import CSVLogger
from comfree_tdmpc.planner.batch_mppi import BatchMPPIPlanner
from comfree_tdmpc.planner.mppi import PlannerConfig
from comfree_tdmpc.reality import apply_reality
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig, load_spec
from comfree_tdmpc.sim.handle_writer import HandleWriter
from comfree_tdmpc.sim.param_space import ParamSpace
from comfree_tdmpc.sysid.stochastic import StochasticSysID, StochasticSysIDConfig


@dataclass
class VecTrainConfig:
    seed: int = 0
    total_env_steps: int = 200_000
    seed_env_steps: int = 2_000
    # Gradient updates per *environment transition* collected.  One update costs
    # 4.6 ms and one vectorised step yields `num_envs` transitions, so the
    # learner has plenty of headroom next to a 640 ms planning call.
    updates_per_env_step: float = 0.5
    max_updates_per_iter: int = 32
    buffer_capacity_per_env: int = 8_000
    sim_buffer_capacity: int = 400_000
    # Fraction of every training batch drawn from harvested MPPI elite rollouts.
    sim_data_ratio: float = 0.5
    # Env steps before the planner is allowed to use the terminal value.  An
    # untrained Q is worse than no Q: the planner maximises into its errors.
    terminal_value_warmup_steps: int = 20_000
    # DAgger-style relabeling: per-env probability of executing the policy's own
    # action instead of the planner's `a`, so the buffer accumulates transitions
    # on states the policy actually visits.  The planner's mu/std are still
    # stored as the imitation label (unaffected by this switch) -- only which
    # action is *executed*, and hence which action is stored for Q-learning,
    # changes.  0.0 = always execute the planner's action.
    dagger_beta: float = 0.0

    # No planner at all: the policy acts in the real env directly every step
    # (plain SAC-style actor-critic). Mutually exclusive with the three
    # switches above and with sysid/dagger/offline, all of which exist only to
    # make planner-collected data better -- see VecTrainer.__init__.
    pure_rl: bool = False

    # SysID schedule, in environment steps.
    sysid_enabled: bool = True
    sysid_every_steps: int = 5_000
    sysid_iters: int = 30
    sysid_warmup_steps: int = 5_000

    # Offline consolidation phase (docs/offline_training_design.md), gated on
    # SysID's val_loss.  None disables the phase.
    offline_start_val_loss: float | None = None
    offline_num_updates: int = 50_000
    offline_batch_size: int = 1024
    offline_value_real_ratio: float = 0.7
    # Require val_loss < offline_start_val_loss on this many consecutive
    # run_sysid() calls before firing -- GPU non-determinism makes a single
    # dip below threshold an unreliable, noise-driven trigger.
    offline_confirm_checks: int = 1
    # Hard floor: never fire before this env step, regardless of val_loss.
    offline_min_step: int | None = None
    # When set, ignore val_loss entirely and fire exactly once as soon as
    # self.step >= offline_force_step -- for controlled experiments that need
    # a fixed, non-noisy trigger point. Takes priority over
    # offline_start_val_loss when both are set.
    offline_force_step: int | None = None
    # "once" = a single one-shot burst.
    # "periodic" = drop the single-fire guard and run a smaller
    # offline_periodic_num_updates burst every time the gate re-qualifies.
    offline_mode: str = "once"
    offline_periodic_num_updates: int = 8_000

    # How far "reality" sits from the compiled model, in normalised u units.
    reality_perturb: float = 0.25
    reality_seed: int = 12345

    eval_every_steps: int = 5_000
    mpc_eval_every_steps: int = 20_000
    video_every_steps: int = 40_000
    save_every_steps: int = 20_000
    log_every_steps: int = 1_000

    out_dir: str = "outputs/vec_run"
    tag: str = ""
    device: str = "cuda:0"

    env: VecEnvConfig = field(default_factory=VecEnvConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    sysid: StochasticSysIDConfig = field(default_factory=StochasticSysIDConfig)


class VecTrainer:
    def __init__(self, cfg: VecTrainConfig):
        self.cfg = cfg
        assert cfg.offline_mode in ("once", "periodic"), cfg.offline_mode
        if cfg.pure_rl:
            assert not cfg.sysid_enabled, (
                "pure_rl: no planner model for SysID to correct")
            assert not cfg.planner.record_elites, (
                "pure_rl: no planner rollouts to harvest into a sim_buffer")
            assert not cfg.planner.use_terminal_value, (
                "pure_rl: no planner rollout for a terminal value to extend")
            assert cfg.dagger_beta == 0.0, (
                "pure_rl: no planner action for dagger_beta to mix with")
            assert cfg.offline_start_val_loss is None and cfg.offline_force_step is None, (
                "pure_rl: the offline phase re-plans against plan_sim, "
                "which pure_rl never builds")
            if cfg.agent.actor_mode != "sac":
                print(f"[trainer] pure_rl forces actor_mode 'sac' "
                      f"(was {cfg.agent.actor_mode!r})", flush=True)
                cfg.agent.actor_mode = "sac"
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.tag = cfg.tag or self.out.name
        self.E = cfg.env.num_envs

        self.space = ParamSpace.realistic(load_spec())
        self.u_model = self.space.true_u.copy()
        self.env = VecAllegroReorientEnv(
            cfg.env, cfg.device, cfg.seed, space=self.space, u=self.u_model
        )
        self.u_reality = apply_reality(
            self.env, self.space, cfg.reality_seed, cfg.reality_perturb
        )
        self.device = self.env.torch_device

        self.agent = TDMPCAgent(cfg.agent, self.env.obs_dim, self.env.action_dim, self.device)
        self.sim_buffer = None
        self.planner = None
        self.plan_sim = None
        if not cfg.pure_rl:
            self.plan_sim = BatchSim(
                SimConfig(
                    nworld=self.E * cfg.planner.num_samples,
                    n_substeps=cfg.env.n_substeps,
                    timestep=cfg.env.timestep,
                    device=cfg.device,
                )
            )
            self.plan_writer = HandleWriter(self.plan_sim, self.space)
            self.plan_writer.write_u(self.u_model)
            if cfg.planner.record_elites:
                self.sim_buffer = ReplayBuffer(
                    cfg.sim_buffer_capacity, self.env.obs_dim, self.env.action_dim, 1, 1, self.device
                )
            self.planner = BatchMPPIPlanner(
                self.plan_sim, self.env.task, cfg.planner, self.E,
                agent=self.agent, sim_buffer=self.sim_buffer,
            )
        self.buffer = VecReplayBuffer(
            cfg.buffer_capacity_per_env, self.E, self.env.obs_dim, self.env.action_dim,
            self.env.sim.nq, self.env.sim.nv, self.device,
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
        self.updates = 0
        self._ran_offline = False
        self._offline_confirm_count = 0
        self._offline_forced = False
        self._last_dagger_frac = 0.0
        self.renderer = None
        self._start = time.time()
        self._recent: list[dict] = []
        self._last = {k: 0 for k in ("log", "eval", "mpc_eval", "video", "save", "sysid")}

    def sample_batch(self) -> dict[str, torch.Tensor]:
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
        # Loss at the true parameters is the floor: what is left above it is
        # model error still on the table.
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

        err = np.abs(self.u_model - self.u_reality)
        after = history[-1].get("val_loss", float("nan")) if history else float("nan")
        row = {
            "step": self.step,
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

    def train_offline(self, num_updates: int) -> None:
        """Offline consolidation phase (docs/offline_training_design.md SS1).

        Plans+harvests against `plan_sim`/`u_model` in a tight loop -- no
        `self.env.step()` -- feeding `sim_buffer` via the existing
        `_store_elites` path, then runs `agent.update_offline()` back-to-back
        with no per-iteration cap, decoupled from the online real-step budget.
        """
        cfg = self.cfg
        assert self.sim_buffer is not None, "train_offline needs --distill (planner.record_elites)"
        print(f"[trainer] offline phase: {num_updates} updates", flush=True)
        qpos, qvel = self.env.state()
        goal_quat, prev_action = self.env.goal_quat, self.env.prev_action
        t0 = self.env.needs_t0.clone()
        n_q_real = int(cfg.offline_batch_size * cfg.offline_value_real_ratio)
        t_start = time.perf_counter()

        for i in range(num_updates):
            self.planner.plan(qpos, qvel, goal_quat, prev_action, t0=t0)
            t0 = torch.zeros_like(t0)  # only the first plan() needs the real warm start

            batch_sim = self.sim_buffer.sample(cfg.offline_batch_size)
            q_real = self.buffer.sample(n_q_real)
            q_sim = self.sim_buffer.sample(cfg.offline_batch_size - n_q_real)
            batch_q = {k: torch.cat([q_real[k], q_sim[k]], dim=0) for k in q_real}
            batch_real = self.buffer.sample(cfg.agent.batch_size)

            self.agent.update_offline(batch_sim, batch_q, batch_real)
            self.updates += 1
            if (i + 1) % max(num_updates // 10, 1) == 0:
                print(f"      [offline] {i + 1}/{num_updates} updates "
                      f"[{time.perf_counter() - t_start:.0f}s]", flush=True)

        print(f"[trainer] offline phase done ({time.perf_counter() - t_start:.0f}s)", flush=True)

    def _maybe_offline(self, s: dict) -> None:
        """Gate + fire the offline phase from a `run_sysid()` result `s`.

        Controlled trigger (docs/offline_training_design.md SS4 extension):
        `offline_force_step` bypasses val_loss entirely and fires once;
        otherwise `offline_confirm_checks` consecutive qualifying
        `run_sysid()` calls are required (GPU non-determinism makes a single
        dip below threshold noise, not a controlled choice), and
        `offline_min_step` is a hard floor on top of either path.
        `offline_mode == "periodic"` drops the single-fire guard and reruns
        a smaller burst every time the gate re-qualifies.
        """
        cfg = self.cfg
        if cfg.offline_start_val_loss is None and cfg.offline_force_step is None:
            return
        if cfg.offline_mode == "once" and self._ran_offline:
            return

        if cfg.offline_force_step is not None:
            if self._offline_forced or self.step < cfg.offline_force_step:
                return
            reason = f"step {self.step} >= offline_force_step {cfg.offline_force_step}"
        else:
            val_loss = s.get("val_loss", float("inf"))
            self._offline_confirm_count = (
                self._offline_confirm_count + 1 if val_loss < cfg.offline_start_val_loss else 0
            )
            if self._offline_confirm_count < cfg.offline_confirm_checks:
                return
            reason = (f"val_loss {val_loss:.4g} < {cfg.offline_start_val_loss} "
                      f"({self._offline_confirm_count}/{cfg.offline_confirm_checks} checks)")

        if cfg.offline_min_step is not None and self.step < cfg.offline_min_step:
            return

        if cfg.offline_force_step is not None:
            self._offline_forced = True
        else:
            self._offline_confirm_count = 0  # allow periodic re-triggering
        self._ran_offline = True
        print(f"[trainer] {reason} -- starting offline phase", flush=True)
        num_updates = (cfg.offline_periodic_num_updates if cfg.offline_mode == "periodic"
                       else cfg.offline_num_updates)
        self.train_offline(num_updates)

    @torch.no_grad()
    def evaluate(self, use_pi: bool, record: bool = False) -> dict:
        """Run one episode in every lane with a deterministic controller."""
        env, cfg = self.env, self.cfg
        env_snap = env.snapshot()
        plan_snap = self.planner.snapshot() if self.planner is not None else None
        env.reset()
        if self.planner is not None:
            self.planner.reset()

        returns = torch.zeros(self.E, device=self.device)
        goals = torch.zeros(self.E, device=self.device)
        drops = torch.zeros(self.E, device=self.device)
        lengths = torch.zeros(self.E, device=self.device)
        live = torch.ones(self.E, dtype=torch.bool, device=self.device)
        frames = []

        obs = env.observation()
        for t in range(cfg.env.episode_length):
            if use_pi:
                a = self.agent.act_batch(obs, eval_mode=True)
            else:
                qpos, qvel = env.state()
                a, _, _ = self.planner.plan(
                    qpos, qvel, env.goal_quat, env.prev_action,
                    t0=env.needs_t0.clone(), eval_mode=True,
                )
            # Lanes that already terminated keep stepping (the vec env auto-reset
            # them) but stop contributing to the statistics.
            next_obs, r, done, info = env.step(a)
            returns += r * live.float()
            lengths += live.float()
            ended = live & done
            goals = torch.where(ended, info["goals_reached"].float(), goals)
            drops = torch.where(ended, info["dropped"], drops)
            if record:
                frames.append(self._frame(info, use_pi, t))
            live = live & ~done
            obs = env.observation()
            if not bool(live.any()):
                break
        # Lanes that never terminated: credit the goals they had accumulated.
        goals = torch.where(live, env.goals_reached.float(), goals)

        env.restore(env_snap)
        if self.planner is not None:
            self.planner.restore(plan_snap)
        out = {
            "return": float(returns.mean().item()),
            "return_std": float(returns.std().item()),
            "goals_reached": float(goals.mean().item()),
            "dropped": float(drops.mean().item()),
            "length": float(lengths.mean().item()),
        }
        if record:
            out["frames"] = frames
        return out

    def _frame(self, info: dict, use_pi: bool, t: int) -> np.ndarray:
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
        at_goal = float(info["success"][0].item()) > 0
        status = (
            f"AT GOAL  holding {int(info['success_streak'][0].item())}"
            f"/{cfg.env.task.success_hold_steps}"
            if at_goal
            else f"goal err={float(info['angle'][0].item()):.2f} rad"
        )
        return self.renderer.frame(
            self.env.qpos[0].cpu().numpy(),
            self.env.goal_quat[0].cpu().numpy(),
            label=f"{self.tag}   {who}\n"
            f"env step {self.step}   t={t}   {status}\n"
            f"goals solved={int(info['goals_reached'][0].item())}   "
            f"model error={theta_err:.3f}",
            at_goal=at_goal,
        )

    def train(self) -> None:
        cfg = self.cfg
        print(f"[trainer] out={self.out}  device={self.device}  num_envs={self.E}")
        print(f"[trainer] {self.space.describe()}")
        if self.planner is not None:
            print(f"[trainer] planner bank = {self.E} x {cfg.planner.num_samples} "
                  f"= {self.plan_sim.nworld} worlds")
        else:
            print("[trainer] pure_rl: no planner -- policy acts in the real env directly")
        print(f"[trainer] reality is {np.abs(self.u_reality - self.u_model).mean():.4f} "
              f"from the compiled model, in normalised units", flush=True)

        obs = self.env.reset()
        if self.planner is not None:
            self.planner.reset()
        update_credit = 0.0

        while self.step < cfg.total_env_steps:
            t_iter = time.perf_counter()
            qpos, qvel = self.env.state()
            seeding = self.step < cfg.seed_env_steps

            if self.planner is not None:
                self.planner.cfg.use_terminal_value = (
                    self._use_terminal_value and self.step >= cfg.terminal_value_warmup_steps
                )
            if seeding:
                # Smoothed noise keeps the cube in the hand long enough to be useful.
                a = (0.75 * self.env.prev_action + 0.4 * torch.randn(
                    self.E, self.env.action_dim, device=self.device)).clamp(-1, 1)
                mu, std = a, torch.full_like(a, cfg.planner.max_std)
            elif cfg.pure_rl:
                # The policy acts alone: sample its own squashed-Gaussian action,
                # same distribution `act_batch` draws from -- mu/std are stored
                # only because the buffer's schema wants them; sac's update_pi
                # never reads either back (see AgentConfig.actor_mode docstring).
                with torch.no_grad():
                    mu, a, _, log_std = self.agent.pi(self.agent.encode(obs))
                std = log_std.exp()
            else:
                a, mu, std = self.planner.plan(
                    qpos, qvel, self.env.goal_quat, self.env.prev_action,
                    t0=self.env.needs_t0.clone(),
                )

            # DAgger-style relabeling: swap in the policy's own action for a
            # per-env-drawn fraction of lanes -- decoupled from `mu`/`std`,
            # which stay the planner's label regardless.
            if seeding or cfg.dagger_beta <= 0.0:
                a_exec = a
                self._last_dagger_frac = 0.0
            else:
                dagger_mask = torch.rand(self.E, device=self.device) < cfg.dagger_beta
                pi_a = self.agent.act_batch(obs)
                a_exec = torch.where(dagger_mask.unsqueeze(-1), pi_a, a)
                self._last_dagger_frac = float(dagger_mask.float().mean().item())

            ep_id = self.env.ep_id.clone()
            next_obs, r, done, info = self.env.step(a_exec)
            self.buffer.add(obs, next_obs, a_exec, mu, std, r, info["dropped"], qpos, qvel, ep_id)
            if bool(done.any()) and self.planner is not None:
                self.planner.reset_idx(done.nonzero(as_tuple=True)[0])
            obs = self.env.observation()
            self.step += self.E
            t_collect = time.perf_counter() - t_iter

            t0 = time.perf_counter()
            metrics = {}
            if not seeding and self.buffer.per_lane * self.E >= cfg.agent.batch_size:
                update_credit += cfg.updates_per_env_step * self.E
                n = min(int(update_credit), cfg.max_updates_per_iter)
                update_credit -= n
                for _ in range(n):
                    metrics = self.agent.update(self.sample_batch())
                    self.updates += 1
            t_learn = time.perf_counter() - t0

            self._recent.extend(self.env.pop_finished())

            if (cfg.sysid_enabled and self.step >= cfg.sysid_warmup_steps
                    and self.step - self._last["sysid"] >= cfg.sysid_every_steps):
                self._last["sysid"] = self.step
                s = self.run_sysid()
                if s:
                    print(f"      [sysid] step={self.step} err {s['param_err_mean']:.4f}  "
                          f"loss {s['loss_before']:.4g} -> {s['val_loss']:.4g} "
                          f"(floor {s['ref_loss']:.4g}, {s['gap_closed']:.0f}% closed) "
                          f"[{s['sec']:.1f}s]", flush=True)
                self._maybe_offline(s)

            if self.step - self._last["log"] >= cfg.log_every_steps:
                self._last["log"] = self.step
                self._log_train(metrics, t_collect, t_learn)

            if self.step - self._last["eval"] >= cfg.eval_every_steps:
                self._last["eval"] = self.step
                self._run_eval()

            if self.step - self._last["save"] >= cfg.save_every_steps:
                self._last["save"] = self.step
                self._save()

        self._run_eval(force_mpc=True)
        self._save()
        self.train_log.save_json(self.out / "config.json", asdict(cfg))
        print(f"[trainer] done: {self.step} env steps, {self.updates} updates, "
              f"{(time.time()-self._start)/60:.1f} min "
              f"({self.step/max(time.time()-self._start,1):.1f} env steps/s)", flush=True)

    def _log_train(self, metrics: dict, t_collect: float, t_learn: float) -> None:
        rec = self._recent[-64:]
        wall = time.time() - self._start
        row = {
            "step": self.step,
            "updates": self.updates,
            "wall_time": wall,
            "env_steps_per_s": self.step / max(wall, 1e-9),
            "episodes": len(self._recent),
            "collect_ms": t_collect * 1e3,
            "learn_ms": t_learn * 1e3,
            "terminal_value_on": float(self.planner.cfg.use_terminal_value) if self.planner else 0.0,
            "dagger_exec_frac": self._last_dagger_frac,
            "sim_buffer": float(len(self.sim_buffer)) if self.sim_buffer else 0.0,
            "param_err_mean": float(np.abs(self.u_model - self.u_reality).mean()),
            "episode_reward": float(np.mean([e["episode_reward"] for e in rec])) if rec else np.nan,
            "episode_length": float(np.mean([e["episode_length"] for e in rec])) if rec else np.nan,
            "goals_reached": float(np.mean([e["goals_reached"] for e in rec])) if rec else np.nan,
            "dropped": float(np.mean([e["dropped"] for e in rec])) if rec else np.nan,
            "mean_angle": float(np.mean([e["mean_angle"] for e in rec])) if rec else np.nan,
            **(self.planner.last_stats if self.planner else {}),
            **metrics,
        }
        self.train_log.log(row)
        print(f"[{self.step:7d}] R={row['episode_reward']:8.1f} "
              f"len={row['episode_length']:5.1f} solved={row['goals_reached']:.2f} "
              f"drop={row['dropped']:.2f} theta_err={row['param_err_mean']:.3f} "
              f"| {row['env_steps_per_s']:5.1f} steps/s  {self.updates} upd "
              f"({t_collect*1e3:.0f}ms plan, {t_learn*1e3:.0f}ms learn)", flush=True)

    def _run_eval(self, force_mpc: bool = False) -> None:
        cfg = self.cfg
        record = (self.step - self._last["video"] >= cfg.video_every_steps) or force_mpc
        do_mpc = self.planner is not None and (
            force_mpc or (self.step - self._last["mpc_eval"] >= cfg.mpc_eval_every_steps)
        )
        if record:
            self._last["video"] = self.step

        ev_pi = self.evaluate(use_pi=True, record=record)
        frames_pi = ev_pi.pop("frames", None)
        erow = {"step": self.step,
                "param_err_mean": float(np.abs(self.u_model - self.u_reality).mean()),
                **{f"pi/{k}": v for k, v in ev_pi.items()}}
        line = f"      [eval] step={self.step} pi R={ev_pi['return']:.1f} " \
               f"solved={ev_pi['goals_reached']:.2f} drop={ev_pi['dropped']:.2f}"

        frames_mpc = None
        if do_mpc:
            self._last["mpc_eval"] = self.step
            ev = self.evaluate(use_pi=False, record=record)
            frames_mpc = ev.pop("frames", None)
            erow.update({f"mpc/{k}": v for k, v in ev.items()})
            line += f" | mpc R={ev['return']:.1f} solved={ev['goals_reached']:.2f}"
        self.eval_log.log(erow)
        print(line, flush=True)

        if frames_pi or frames_mpc:
            from comfree_tdmpc.render import save_gif

            if frames_pi:
                save_gif(frames_pi, self.out / "videos" / f"s{self.step:07d}_pi.gif", fps=25)
            if frames_mpc:
                save_gif(frames_mpc, self.out / "videos" / f"s{self.step:07d}_mpc.gif", fps=25)

    def _save(self) -> None:
        self.agent.save(str(self.out / "agent.pt"))
        np.save(self.out / "params.npy", self.u_model)
        np.save(self.out / "params_reality.npy", self.u_reality)
