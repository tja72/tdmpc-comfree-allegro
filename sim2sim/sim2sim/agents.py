"""Load trained agents with the config they were trained with.

comfree_tdmpc: `AgentConfig(**config.json["agent"])`, `PlannerConfig(**config.json
["planner"])` (never the dataclass defaults, which differ from what the runs
were trained with).  The planner keeps its own ComFree GPU bank holding `params.npy`,
the u_model it planned with at the end of training (SysID estimate, or the
nominal model if SysID was off) -- no information from the MuJoCo plant.

tdmpc-square-pearl: the Hydra config of the training run
(`outputs/<date>/<time>/.hydra/config.yaml`), run through pearl's own
`parse_cfg` and `make_env`, then `TDMPC2(cfg).load(ckpt)`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.agent.tdmpc_agent import AgentConfig, TDMPCAgent
from comfree_tdmpc.envs.task import ReorientTask
from comfree_tdmpc.planner.batch_mppi import BatchMPPIPlanner
from comfree_tdmpc.planner.mppi import PlannerConfig
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig
from comfree_tdmpc.sim.handle_writer import HandleWriter
from comfree_tdmpc.sim.param_space import ParamSpace

WORKSPACE = Path(__file__).resolve().parents[2]
PEARL_ROOT = WORKSPACE / "tdmpc-square-pearl"


def load_run_config(run: Path) -> dict:
    return json.loads((run / "config.json").read_text())


class ComfreeController:
    """comfree_tdmpc agent acting in a VecAllegroReorientEnv-like env.

    mode="pi":  `agent.act_batch(obs, eval_mode=True)` (VecTrainer.evaluate use_pi).
    mode="mpc": `planner.plan(qpos, qvel, goal, prev_action, t0, eval_mode=True)` with
                the state read from whichever env is being evaluated -- for the
                MuJoCo env that *is* the state sync (ComFree state == (qpos, qvel)).
    """

    def __init__(self, run: Path, mode: str, num_envs: int, task: ReorientTask,
                 space: ParamSpace, num_samples: int | None = None):
        assert mode in ("pi", "mpc")
        self.run, self.mode, self.E = Path(run), mode, num_envs
        self.config = cfg = load_run_config(self.run)
        self.device = task.device
        self.agent_cfg = AgentConfig(**cfg["agent"])
        self.agent = TDMPCAgent(self.agent_cfg, task.obs_dim, 16, self.device)
        self.agent.load(str(self.run / "agent.pt"))
        self.agent.eval()
        self.planner = None
        self.u_model = None
        if mode == "mpc":
            if cfg.get("pure_rl", False) or "planner" not in cfg:
                raise ValueError(f"{run} was trained without a planner (pure_rl)")
            pc = PlannerConfig(**cfg["planner"])
            pc.record_elites = False
            # VecTrainer switches the terminal value on after its warmup; at the end
            # of training that is the config value unless the run was shorter.
            pc.use_terminal_value = bool(
                cfg["planner"]["use_terminal_value"]
                and cfg.get("total_env_steps", 0) >= cfg.get("terminal_value_warmup_steps", 20_000)
            )
            if num_samples is not None:
                pc.num_samples = num_samples
                pc.num_elites = min(pc.num_elites, num_samples)
                pc.num_pi_trajs = min(pc.num_pi_trajs, num_samples)
            self.planner_cfg = pc
            p = self.run / "params.npy"
            self.u_model = np.load(p) if p.exists() else space.true_u.copy()
            sim = BatchSim(SimConfig(
                nworld=num_envs * pc.num_samples,
                n_substeps=cfg["env"]["n_substeps"],
                timestep=cfg["env"]["timestep"],
                device="cuda:0",
            ))
            HandleWriter(sim, space).write_u(self.u_model)
            self.planner = BatchMPPIPlanner(sim, task, pc, num_envs, agent=self.agent)

    def describe(self) -> dict:
        d = {"run": str(self.run), "mode": self.mode, "agent": self.config["agent"]}
        if self.planner is not None:
            d["planner"] = vars(self.planner_cfg)
            d["planner_params"] = "params.npy" if (self.run / "params.npy").exists() else "nominal true_u"
            d["sysid_enabled"] = self.config.get("sysid_enabled")
        return d

    def reset(self) -> None:
        if self.planner is not None:
            self.planner.reset()

    @torch.no_grad()
    def act(self, env) -> torch.Tensor:
        if self.mode == "pi":
            return self.agent.act_batch(env.observation(), eval_mode=True)
        qpos, qvel = env.state()
        a, _, _ = self.planner.plan(
            qpos.to(self.device), qvel.to(self.device), env.goal_quat, env.prev_action,
            t0=env.needs_t0.clone(), eval_mode=True,
        )
        return a


def find_hydra_dir(log_dir: Path) -> Path:
    """The Hydra run dir whose overrides match this log dir's task/seed/exp_name."""
    log_dir = Path(log_dir).resolve()
    exp_name, seed, task = log_dir.name, log_dir.parent.name, log_dir.parent.parent.name
    want = {f"task={task}", f"seed={seed}", f"exp_name={exp_name}"}
    hits = []
    for ov in sorted((PEARL_ROOT / "outputs").glob("*/*/.hydra/overrides.yaml")):
        lines = {ln.strip().lstrip("- ").strip() for ln in ov.read_text().splitlines()}
        if want <= lines:
            hits.append(ov.parent.parent)
    if not hits:
        raise FileNotFoundError(f"no Hydra run dir in {PEARL_ROOT/'outputs'} with overrides {want}; "
                                f"pass --pearl_hydra_dir")
    if len(hits) > 1:
        print(f"[sim2sim] {len(hits)} Hydra dirs match {want}; using the latest {hits[-1]}")
    return hits[-1]


def load_pearl(log_dir: Path, ckpt: str = "final.pt", hydra_dir: Path | None = None):
    """Returns (cfg, gym_env, agent, hydra_dir); gym_env is TensorWrapper(ComFreeAllegroEnv)."""
    import hydra.utils
    from omegaconf import OmegaConf

    from tdmpc_square.common.parser import parse_cfg
    from tdmpc_square.envs import make_env
    from tdmpc_square.tdmpc_square import TDMPC2

    hydra_dir = Path(hydra_dir) if hydra_dir else find_hydra_dir(log_dir)
    cfg = OmegaConf.load(hydra_dir / ".hydra" / "config.yaml")
    # parse_cfg asks Hydra for the launch cwd, which only exists inside
    # @hydra.main; pearl expects it to be its repo root.
    orig = hydra.utils.get_original_cwd
    hydra.utils.get_original_cwd = lambda: str(PEARL_ROOT)
    try:
        cfg = parse_cfg(cfg)
    finally:
        hydra.utils.get_original_cwd = orig
    assert cfg.task == "comfree-allegro-cube", cfg.task
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    agent.load(str(Path(log_dir) / "models" / ckpt))
    return cfg, env, agent, hydra_dir
