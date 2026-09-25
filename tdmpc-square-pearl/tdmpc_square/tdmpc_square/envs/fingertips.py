"""
Fingertip manipulation tasks: three point-fingertips push/lift/flip a free object
toward a target pose.

Physics and per-object/variant task setup are ported from the "Fingertips manipulation"
tasks of Complementarity-Free Multi-Contact Modeling and Optimization for Dexterous
Manipulation (Jin, 2024) - https://arxiv.org/abs/2408.07855
(https://github.com/asu-iris/Complementarity-Free-Dexterous-Manipulation).

That repo drives these tasks with a classical contact-implicit MPC solver (IPOPT via
CasADi) and has no notion of a per-step reward or episode. Here the raw MuJoCo
simulator (`envs/fingertips_env.py::MjSimulator` there) is re-implemented headless
(no live viewer in the step hot path) and wrapped with a reward/termination/observation
scheme so it can be driven by TD-MPC2's CEM planner instead.

Not a `gymnasium.Env` subclass: `tdmpc_square` only needs `reset()`/`step()` (5-tuple),
`.observation_space`, `.action_space` and `.max_episode_steps` (see
`envs/wrappers/tensor.py`).

Task string convention: "fingertips-<object>-<variant>", e.g. "fingertips-cube-air".
Valid (object, variant) pairs (mirrors the original repo's README table):
    cube:      air, ground_flip, ground_rotation
    bunny:     air, ground_rotation
    foambrick: air, ground_flip, ground_rotation
    stick:     ground_flip

MuJoCo assets live in `envs/fingertips_assets/`: a shared `fingertips.xml` holds
everything common to all four objects; each object has its own subfolder
(`<object>/<object>.xml`, split by `<!-- ASSET -->` / `<!-- BODY -->` markers) plus its
mesh/texture file(s). `_load_model()` splices the fragment into the shared base and
compiles via `mujoco.MjModel.from_xml_string`. Reward terms live in
`fingertips_assets/rewards.py`; tunables/weights in `fingertips_assets/fingertips.yaml`,
merged with `cfg.fingertips` by `make_env()`, e.g.:
    python -m tdmpc_square.train task=fingertips-cube-air +fingertips.reward.style=shaped
The directory is named `fingertips_assets`, not `fingertips`, to avoid a package/module
name collision with this file.
"""

import os

import gymnasium as gym
import mujoco
import numpy as np
from omegaconf import OmegaConf

from tdmpc_square.envs.fingertips_assets import rewards as rewards

# xml paths/markers
_ASSET_DIR = os.path.join(os.path.dirname(__file__), "fingertips_assets")
_BASE_XML_PATH = os.path.join(_ASSET_DIR, "fingertips.xml")
_ASSET_MARKER = "<!-- OBJECT_ASSET -->"
_BODY_MARKER = "<!-- OBJECT_BODY -->"
_DEFAULTS_PATH = os.path.join(_ASSET_DIR, "fingertips.yaml")

# rotation helpers (sampling only -- reward-related error/rotation math is in rewards.py)


def _rpy_to_quat(yaw, pitch, roll):
    qx = (np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2)
          - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2))
    qy = (np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
          + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2))
    qz = (np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
          - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2))
    qw = (np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2)
          + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2))
    return np.array([qw, qx, qy, qz])


def _axisangle_to_quat(axis, angle):
    axis = axis / np.linalg.norm(axis)
    quat = np.zeros(4)
    quat[0] = np.cos(angle / 2)
    quat[1:] = np.sin(angle / 2) * axis
    return quat


# Configs per-object / per-variant task table, ported from Complementarity-Free-Dexterous-Manipulation/examples/mpc/fingertips/*/params.py
_OBJECTS = {
    "cube": dict(
        assets=["iris_block.png"], init_height=0.03,
        variants={
            "ground_rotation": dict(height=0.03, pitch=False, roll=False),
            "ground_flip": dict(height=0.05, pitch=True, roll=True),
            "air": dict(height_range=(0.05, 0.10)),
        },
    ),
    "bunny": dict(
        assets=["bunny.stl"], init_height=0.021,
        variants={
            "ground_rotation": dict(height=0.021, pitch=False, roll=False),
            "air": dict(height_range=(0.03, 0.08)),
        },
    ),
    "foambrick": dict(
        assets=["foam_brick.stl", "general_block.png"], init_height=0.03,
        variants={
            "ground_rotation": dict(height=0.03, pitch=False, roll=False),
            "ground_flip": dict(height=0.03, pitch=True, roll=True),
            "air": dict(height_range=(0.04, 0.09)),
        },
    ),
    "stick": dict(
        assets=["stick.stl", "oak.png"], init_height=0.015,
        variants={
            "ground_flip": dict(height=0.015, pitch=False, roll=True),
        },
    ),
}

def _load_defaults(overrides=None):
    """Load fingertips.yaml and merge `overrides` (e.g. cfg.fingertips) over it."""
    defaults = OmegaConf.load(_DEFAULTS_PATH)
    if overrides is None:
        return defaults
    return OmegaConf.merge(defaults, overrides)


def _sample_init_obj_qpos(cfg, rng):
    xy = 0.05 * rng.random(2) - 0.025
    yaw = 2 * np.pi * rng.random() - np.pi
    quat = _rpy_to_quat(yaw, 0.0, 0.0)
    return np.hstack([xy, cfg["init_height"], quat])


def _sample_target(cfg, variant, rng):
    v = cfg["variants"][variant]
    xy = 0.2 * rng.random(2) - 0.1
    if "height_range" in v:  # air (in-air manipulation)
        lo, hi = v["height_range"]
        height = lo + (hi - lo) * rng.random()
        target_pos = np.hstack([xy, height])
        angle = 2 * np.pi * rng.random() - np.pi
        axis = np.array([0.0, 1.0, 1.0]) + 0.1 * rng.standard_normal(3)
        target_quat = _axisangle_to_quat(axis, angle)
    else:  # ground_rotation / ground_flip
        target_pos = np.hstack([xy, v["height"]])
        yaw = 2 * np.pi * rng.random() - np.pi
        pitch = (np.pi * rng.random() - np.pi / 2) if v["pitch"] else 0.0
        roll = (np.pi * rng.random() - np.pi / 2) if v["roll"] else 0.0
        target_quat = _rpy_to_quat(yaw, pitch, roll)
    return target_pos, target_quat


def _load_model(object_name, cfg):
    """Splice the shared fingertips.xml base with this object's fragment and compile."""
    object_dir = os.path.join(_ASSET_DIR, object_name)
    fragment_path = os.path.join(object_dir, f"{object_name}.xml")
    if not os.path.isfile(fragment_path):
        raise FileNotFoundError(
            f"Fingertips MuJoCo fragment not found at '{fragment_path}'. Expected "
            f"envs/fingertips_assets/{object_name}/{object_name}.xml plus its mesh/texture "
            f"file(s) in the same folder."
        )

    # combine xml of fingertips with xml of object on the run
    fragment = open(fragment_path).read()
    _, _, rest = fragment.partition("<!-- ASSET -->")
    asset_xml, _, body_xml = rest.partition("<!-- BODY -->")

    base_xml = open(_BASE_XML_PATH).read()
    combined_xml = base_xml.replace(_ASSET_MARKER, asset_xml.strip())
    combined_xml = combined_xml.replace(_BODY_MARKER, body_xml.strip())

    assets = {
        fname: open(os.path.join(object_dir, fname), "rb").read()
        for fname in cfg["assets"]
    }
    return mujoco.MjModel.from_xml_string(combined_xml, assets)


class FingertipsEnv:
    """
    Three point-fingertips manipulate a free object toward a target pose.

    Observation: [qpos(16), qvel(15), fingertip-object relative vectors(9), pos_err
    vector(3), relative rotation quat(4)], optionally with [target_pos(3), target_quat(4)]
    appended when `goal_conditioned=True` (only useful once the target itself varies).
    Action: fingertip position-delta command, shape (9,), scaled from [-1, 1] to the
    underlying PD controller's raw [-0.005, 0.005] meter range.
    Reward: selected by `cfg.reward.style`, "legacy" or "shaped" (see
    fingertips_assets/fingertips.yaml and fingertips_assets/rewards.py for the terms).
    `info["success"]` is a separate threshold check (pos_error/quat_error vs.
    success_*_threshold), not derived from either reward style.

    `cfg`, if given, overrides fingertips.yaml's defaults.
    """

    def __init__(
        self,
        object_name,
        variant,
        goal_conditioned=True,
        cfg=None,
        seed=None,
    ):
        if object_name not in _OBJECTS:
            raise ValueError(
                f"Unknown fingertips object '{object_name}'; available: {sorted(_OBJECTS)}"
            )
        task = _OBJECTS[object_name]
        if variant not in task["variants"]:
            raise ValueError(
                f"Object '{object_name}' has no variant '{variant}'; "
                f"available: {sorted(task['variants'])}"
            )

        self.cfg = _load_defaults(cfg)
        self.max_episode_steps = self.cfg.episode.max_episode_steps

        self._task = task
        self._object_name = object_name
        self._variant = variant
        self._goal_conditioned = goal_conditioned
        self._rng = np.random.default_rng(seed)

        self.model = _load_model(object_name, task)
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        self._renderer_size = None

        self._fixed_target_pos, self._fixed_target_quat = _sample_target(task, variant, self._rng)
        self._target_pos = self._fixed_target_pos.copy()
        self._target_quat = self._fixed_target_quat.copy()
        self._elapsed_steps = 0

        # +9: fingertip-object relative vectors. +7: pos_err vector(3) + relative rotation
        # quat(4) -- these save the encoder from having to learn obj_pos - target_pos /
        # relative rotation from scratch, useful even with a fixed target since the target
        # itself is constant but the error against it isn't. +7 goal-conditioned: raw
        # target_pos(3) + target_quat(4), only informative once the target varies per episode
        obs_dim = self.cfg.sim.n_qpos + self.cfg.sim.n_qvel + 9 + 7 + (7 if goal_conditioned else 0)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.sim.n_cmd,), dtype=np.float32
        )

        # without this, data.qpos/qvel stay at MuJoCo's raw defaults and the goal body
        # stays at the XML's static placeholder until the first explicit reset() call
        self.reset()

    def _set_goal_body(self):
        self.model.body("goal").pos = self._target_pos
        self.model.body("goal").quat = self._target_quat

    def _get_obs(self):
        obj_pos = self.data.qpos[0:3]
        obj_quat = self.data.qpos[3:7]
        ft_pos = self.data.qpos[7:].reshape(3, 3)
        ft_rel = (ft_pos - obj_pos[None, :]).reshape(-1)  # fingertip-to-object relative vectors
        pos_err_vec = obj_pos - self._target_pos
        quat_rel = rewards.quat_mul(rewards.quat_conj(self._target_quat), obj_quat)

        obs = np.concatenate(
            [self.data.qpos, self.data.qvel, ft_rel, pos_err_vec, quat_rel]
        ).astype(np.float32)
        if self._goal_conditioned:
            obs = np.concatenate([obs, self._target_pos, self._target_quat]).astype(np.float32)
        return obs

    def render(self, mode="rgb_array", width=384, height=384, camera_id=0):
        if self._renderer is None or self._renderer_size != (height, width):
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer_size = (height, width)
        self._renderer.update_scene(self.data, camera=camera_id)
        return self._renderer.render()

    def _errors(self):
        pos_err = rewards.pos_error(self.data.qpos[0:3], self._target_pos)
        quat_err = rewards.quat_error(self.data.qpos[3:7], self._target_quat)
        return pos_err, quat_err

    def _cost(self, cmd):
        """style: legacy -- ported MPC cost, see fingertips_assets/fingertips.yaml."""
        obj_pos = self.data.qpos[0:3]
        obj_quat = self.data.qpos[3:7]
        ft_pos = self.data.qpos[7:].reshape(3, 3)
        dcm = rewards.quat_to_dcm(obj_quat)

        c = self.cfg.reward.legacy
        return (
            c.contact_cost_weight * rewards.contact_cost(obj_pos, ft_pos)
            + c.grasp_closure_weight * rewards.grasp_closure(obj_pos, dcm, ft_pos)
            + c.control_cost_weight * rewards.control_cost(cmd)
            + c.position_cost_weight * rewards.position_cost(obj_pos, self._target_pos)
            + c.quaternion_cost_weight * rewards.quaternion_cost(obj_quat, self._target_quat)
        )

    def _reward_shaped(self, cmd, pos_err, quat_err):
        """style: shaped -- see fingertips_assets/fingertips.yaml."""
        obj_pos = self.data.qpos[0:3]
        obj_quat = self.data.qpos[3:7]
        ft_pos = self.data.qpos[7:].reshape(3, 3)
        dcm = rewards.quat_to_dcm(obj_quat)
        s = self.cfg.reward.shaped

        # contact_cost - (good when 1)
        finger_obj_distance_tanh = s.finger_obj_distance.weight * rewards.finger_obj_distance_tanh(obj_pos, ft_pos, s.finger_obj_distance.std)

        # control_cost - (good when 0)
        ctrl_magnitude_square = s.ctrl_magnitude.weight * rewards.control_cost(cmd)

        # grasp_closure - (good when 1)
        grasp_distribution_exp = s.grasp_distribution.weight * rewards.grasp_distribution_exp(obj_pos, dcm, ft_pos, s.grasp_distribution.std, s.grasp_distribution.gate_std)

        # position_cost - absolute (rewards holding at the goal), gated by continuous
        # engagement (worst-fingertip distance) instead of boolean "any contact", so a
        # 2-close-1-idle grasp can't collect the same reward as genuine 3-way engagement.
        # Uses its own tight tracking_gate_std (near-contact scale, like
        # grasp_distribution.gate_std) rather than finger_obj_distance.std -- that one is
        # deliberately wide for long-range approach shaping, which would make this gate far
        # too lenient to a distant idle fingertip
        track_obj_pos_tanh_gated = s.position_tracking.weight * rewards.track_obj_pos_tanh_gated(pos_err, obj_pos, ft_pos, s.position_tracking.std, s.tracking_gate_std)
        # position_cost - progress (rewards closing the error, not lucky starts), never gated,
        # own (wider) std so real progress made far from the goal isn't discounted as heavily
        track_obj_pos_progress = s.position_tracking.progress_weight * rewards.position_progress(pos_err, self._prev_pos_err, s.position_tracking.progress_std)

        # quaterion_cost - absolute, same tight engagement gate
        track_obj_quad_tanh_gated = s.quaternion_tracking.weight * rewards.track_obj_quad_tanh_gated(quat_err, obj_pos, ft_pos, s.quaternion_tracking.std, s.tracking_gate_std)
        # quaterion_cost - progress, never gated, own (wider) std
        track_obj_quad_progress = s.quaternion_tracking.progress_weight * rewards.quaternion_progress(quat_err, self._prev_quat_err, s.quaternion_tracking.progress_std)

        return float(
            finger_obj_distance_tanh + grasp_distribution_exp + ctrl_magnitude_square
            + track_obj_pos_tanh_gated + track_obj_pos_progress + track_obj_quad_tanh_gated + track_obj_quad_progress
        )

    def _is_success(self, pos_err, quat_err):
        return (
            pos_err < self.cfg.episode.success_pos_threshold
            and quat_err < self.cfg.episode.success_quat_threshold
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # either fixed or random target
        if self._goal_conditioned:
            self._target_pos, self._target_quat = _sample_target(self._task, self._variant, self._rng)
        else:
            self._target_pos = self._fixed_target_pos.copy()
            self._target_quat = self._fixed_target_quat.copy()
        self._set_goal_body()

        init_obj_qpos = _sample_init_obj_qpos(self._task, self._rng)
        self.data.qpos[:] = np.hstack([init_obj_qpos, self.cfg.sim.init_robot_qpos])
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._elapsed_steps = 0

        pos_err, quat_err = self._errors()
        self._prev_pos_err, self._prev_quat_err = pos_err, quat_err  # progress reward baseline
        success = self._is_success(pos_err, quat_err)
        info = dict(
            success=float(success),
            success_subtasks=float(success),
            pos_error=pos_err,
            quat_error=quat_err,
            target_pos=self._target_pos.copy(),
            target_quat=self._target_quat.copy(),
        )
        return self._get_obs(), info

    def step(self, action):
        cmd = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) * self.cfg.sim.action_bound

        # fingertip mass matrix, for gravity compensation in the PD loop below
        n_qvel, n_cmd = self.cfg.sim.n_qvel, self.cfg.sim.n_cmd
        full_m = np.zeros((n_qvel, n_qvel), dtype=np.float64, order="C")
        mujoco.mj_fullM(self.model, full_m, self.data.qM)
        fingertip_m = full_m[-n_cmd:, -n_cmd:]

        desired_fts_pos = self.data.qpos[7:].copy() + cmd

        # 50 Hz control, PD loop at the sim rate (frame_skip substeps)
        for _ in range(self.cfg.sim.frame_skip):
            dpos = self.data.qpos[7:] - desired_fts_pos
            dvel = self.data.qvel[6:]
            self.data.ctrl[:] = (
                -100 * dpos - 2 * dvel - fingertip_m @ np.tile(self.model.opt.gravity, 3)
            )
            mujoco.mj_step(self.model, self.data, nstep=1)


        self._elapsed_steps += 1
        pos_err, quat_err = self._errors()
        success = self._is_success(pos_err, quat_err)
        if self.cfg.reward.style == "shaped":
            reward = self._reward_shaped(cmd, pos_err, quat_err)
        elif self.cfg.reward.style == "legacy":
            reward = (self.cfg.reward.legacy.baseline - self._cost(cmd)) / self.cfg.reward.legacy.scale
        else:
            raise ValueError(
                f"Unknown fingertips.reward.style '{self.cfg.reward.style}'; "
                f"expected 'legacy' or 'shaped'"
            )
        self._prev_pos_err, self._prev_quat_err = pos_err, quat_err  # progress reward baseline

        # never terminate early on success: the object can still drift off target, and
        # a hard terminal state would change how TD-MPC2 bootstraps Q past this step
        terminated = False
        truncated = self._elapsed_steps >= self.max_episode_steps
        info = dict(
            success=float(success),
            success_subtasks=float(success),
            pos_error=pos_err,
            quat_error=quat_err,
            target_pos=self._target_pos.copy(),
            target_quat=self._target_quat.copy(),
        )
        return self._get_obs(), reward, terminated, truncated, info


def make_env(cfg):
    """
    Make a fingertip manipulation environment.
    Task string: "fingertips-<object>-<variant>", e.g. "fingertips-cube-air".
    """
    if not cfg.task.startswith("fingertips"):
        raise ValueError(f"Task {cfg.task} not supported")
    try:
        _, object_name, variant = cfg.task.split("-", 2)
    except ValueError:
        raise ValueError(
            f"Task {cfg.task} not supported; expected fingertips-<object>-<variant>"
        )

    env = FingertipsEnv(
        object_name=object_name,
        variant=variant,
        goal_conditioned=cfg.get("goal_conditioned", True),
        cfg=cfg.get("fingertips", None),
        seed=cfg.get("seed", None),
    )
    return env
