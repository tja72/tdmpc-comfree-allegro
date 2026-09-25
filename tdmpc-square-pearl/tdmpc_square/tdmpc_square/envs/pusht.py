import gymnasium as gym
import gym_pusht  # noqa: F401  registers gym_pusht/PushT-v0

from gymnasium.wrappers import RescaleAction
import numpy as np

PIXEL_SCALE = 512.0  # side length of the pusht render/coordinate space
VEL_NORM = 2500.0
VELOCITY_PENALTY_ALPHA = 0.1
DELTA_ACTION_MAX = 50.0  # max pixel displacement per step when delta_actions is enabled

class PushTInfoWrapper(gym.Wrapper):
    """Adds the info["success"]/["success_subtasks"] keys tdmpc_square expects."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["success"] = float(info.get('is_success', terminated))  # terminated=True means task solved
        info["success_subtasks"] = info["success"]
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class RegularizeMovements(gym.Wrapper):
    """Penalizes fast agent movement: reward -= alpha * |normalized velocity|^2."""
    def __init__(self, env, velocity_penalty_alpha: float = VELOCITY_PENALTY_ALPHA):
        super().__init__(env)
        print(f"Using velocity penalty alpha: {velocity_penalty_alpha}")
        self._alpha = velocity_penalty_alpha
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # obs[5:7] = normalized velocity appended by AppendVelocityObs below us
        vel_norm = obs[5:7]
        accel_penalty = self._alpha * float(np.dot(vel_norm, vel_norm))
        reward = reward - accel_penalty


        info["task_reward"] = float(reward + accel_penalty)
        info["accel_penalty"] = float(accel_penalty)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


def _delta_action_speed_factor(env, iters: int = 20) -> float:
    """
    Calibrates the steady-state ratio between a constant per-step delta and the
    resulting average actuator speed (raw px/s), by replaying the exact PD-control
    recurrence the underlying PushTEnv uses internally (k_p, k_v, dt, control_hz).
    The recurrence is linear, so this ratio is independent of the delta magnitude
    used to calibrate it -- i.e. steady_speed = factor * delta_action_max always.
    """
    base = env.unwrapped
    k_p, k_v, dt = base.k_p, base.k_v, base.dt
    n_substeps = int(1 / (dt * base.control_hz))
    step_duration = n_substeps * dt

    x, v, delta, prev_x = 0.0, 0.0, 1.0, 0.0
    for _ in range(iters):
        prev_x = x
        target = x + delta
        for _ in range(n_substeps):
            accel = k_p * (target - x) - k_v * v
            v += accel * dt
            x += v * dt
    return (x - prev_x) / step_duration


class DeltaActionWrapper(gym.Wrapper):
    """
    Converts delta actions (a per-step displacement relative to the agent's current
    position) into the absolute pixel-target actions expected by the underlying PD
    controller. Exposes a [-1, 1] action space; +-1 corresponds to +-max_delta pixels.
    Must wrap the raw env directly, before any observation normalization, since it
    relies on the raw pixel agent position in obs[:2].

    If `max_speed` (raw px/s) is given, it takes precedence over `max_delta`: it
    caps the agent's actual steady-state physical speed, not the action signal
    itself -- the equivalent `max_delta` is derived via `_delta_action_speed_factor`.
    """

    def __init__(self, env, max_delta: float = DELTA_ACTION_MAX, max_speed: float = None):
        super().__init__(env)
        if max_speed is not None:
            factor = _delta_action_speed_factor(env)
            max_delta = max_speed / factor
            print(f"[DeltaActionWrapper] max_speed={max_speed} px/s -> calibrated max_delta={max_delta:.2f} px/step")
        self._max_delta = max_delta
        self._action_low = env.action_space.low.copy()
        self._action_high = env.action_space.high.copy()
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32
        )
        self._agent_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._agent_pos = obs[:2].astype(np.float32).copy()
        return obs, info

    def step(self, action):
        delta = np.asarray(action, dtype=np.float32) * self._max_delta
        target = np.clip(self._agent_pos + delta, self._action_low, self._action_high)
        obs, reward, terminated, truncated, info = self.env.step(target)
        self._agent_pos = obs[:2].astype(np.float32).copy()
        return obs, reward, terminated, truncated, info


class NormalizePushTObs(gym.ObservationWrapper):
    """Normalise all obs dimensions to [0, 1]."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

    def observation(self, obs):
        obs = obs.astype(np.float32)
        obs[:4] /= PIXEL_SCALE    # agent xy + block xy: [0,512] → [0,1]
        obs[4] /= (2 * np.pi)     # block angle: [0,2π] → [0,1]
        np.clip(obs, 0.0, 1.0, out=obs)
        return obs

class AppendVelocityObs(gym.Wrapper):
    """Appends info["vel_agent"], normalized to [-1, 1], to the observation.
    Must be applied after NormalizePushTObs."""

    def __init__(self, env):
        super().__init__(env)
        low = np.concatenate([env.observation_space.low, [-1.0, -1.0]]).astype(np.float32)
        high = np.concatenate([env.observation_space.high, [1.0, 1.0]]).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        vel_norm = np.array(info["vel_agent"], dtype=np.float32) / VEL_NORM
        np.clip(vel_norm, -1.0, 1.0, out=vel_norm)
        obs = np.concatenate([obs, vel_norm])
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        vel_raw = np.array(info["vel_agent"], dtype=np.float32)
        vel_norm = np.clip(vel_raw / VEL_NORM, -1.0, 1.0)
        obs = np.concatenate([obs, vel_norm])
        return obs, info


def make_env(cfg):
    """
    Make PushT environment.
    Adapted from https://github.com/huggingface/gym-pusht

        python -m tdmpc_square.train task=pusht delta_actions=true delta_action_max=50.0 horizon=5 model_size=5
    """

    if not cfg.task.startswith("pusht"):
        raise ValueError(f"Task {cfg.task} not supported")

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    max_episode_steps = env._max_episode_steps

    # optionally convert to delta (relative-displacement) actions for smoother motion
    if cfg.get("delta_actions", False):
        env = DeltaActionWrapper(
            env,
            max_delta=cfg.get("delta_action_max", DELTA_ACTION_MAX),
            max_speed=cfg.get("max_speed", None),
        )
        # Local deltas make long-range moves harder to plan; may need a longer horizon.

    env = NormalizePushTObs(env)
    # velocity in the obs keeps the problem Markovian under the velocity penalty
    env = AppendVelocityObs(env)
    env = PushTInfoWrapper(env)
    env = RegularizeMovements(env, velocity_penalty_alpha=cfg.get('velocity_penalty_alpha', VELOCITY_PENALTY_ALPHA))
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)

    env.max_episode_steps = max_episode_steps

    return env
