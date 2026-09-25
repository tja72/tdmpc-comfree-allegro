"""Offscreen rendering and gif export.

The renderer uses its own copy of the scene compiled *with* a translucent goal
marker.  The marker is a mocap body, so it adds no degrees of freedom and the
render model stays state-compatible with the physics model.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Must be set before the first renderer is created.  EGL is the headless path
# that works under WSL2; OSMesa is the software fallback.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from comfree_tdmpc.sim.batch_sim import load_spec  # noqa: E402


class SceneRenderer:
    def __init__(
        self,
        width: int = 480,
        height: int = 360,
        camera: str = "demo-cam",
        show_goal: bool = True,
        goal_pos: tuple[float, float, float] = (0.145, -0.045, 0.155),
    ):
        self.mjm = load_spec(add_goal_ghost=True)
        self.mjd = mujoco.MjData(self.mjm)
        self.renderer = mujoco.Renderer(self.mjm, height=height, width=width)
        self.camera = camera
        self.show_goal = show_goal
        self.goal_pos = np.array(goal_pos, dtype=np.float64)
        self.nq = self.mjm.nq
        self.nv = self.mjm.nv
        gid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_GEOM, "goal_ghost_geom")
        self._ghost_geom = gid if gid >= 0 else None

    def frame(
        self,
        qpos: np.ndarray,
        goal_quat: np.ndarray | None = None,
        label: str | None = None,
        at_goal: bool = False,
    ) -> np.ndarray:
        self.mjd.qpos[: self.nq] = np.asarray(qpos, dtype=np.float64)[: self.nq]
        self.mjd.qvel[:] = 0.0
        if self.show_goal and self.mjm.nmocap > 0:
            self.mjd.mocap_pos[0] = self.goal_pos
            if goal_quat is not None:
                self.mjd.mocap_quat[0] = np.asarray(goal_quat, dtype=np.float64)
            if self._ghost_geom is not None:
                # Green while the cube is inside the success cone.
                self.mjm.geom_rgba[self._ghost_geom] = (
                    (0.25, 1.0, 0.35, 0.75) if at_goal else (1.0, 1.0, 1.0, 0.45)
                )
        mujoco.mj_forward(self.mjm, self.mjd)
        self.renderer.update_scene(self.mjd, camera=self.camera)
        img = self.renderer.render()
        if label:
            img = draw_label(img, label)
        return img

    def close(self) -> None:
        self.renderer.close()


_FONT_CACHE: dict[int, object] = {}


def _font(size: int = 14):
    """A real TTF (matplotlib ships DejaVu); PIL's builtin bitmap font is unreadable."""
    if size not in _FONT_CACHE:
        from PIL import ImageFont

        font = None
        try:
            import matplotlib

            path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
            if path.exists():
                font = ImageFont.truetype(str(path), size)
        except Exception:  # noqa: BLE001
            font = None
        _FONT_CACHE[size] = font or ImageFont.load_default()
    return _FONT_CACHE[size]


def draw_label(img: np.ndarray, text: str, size: int = 14) -> np.ndarray:
    """Burn a caption into the top-left corner of a frame."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return img
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, "RGBA")
    font = _font(size)
    lines = text.split("\n")
    pad, lh = 5, size + 4
    width = max(draw.textlength(line, font=font) for line in lines)
    draw.rectangle([0, 0, width + 2 * pad, lh * len(lines) + 2 * pad], fill=(0, 0, 0, 205))
    for i, line in enumerate(lines):
        draw.text((pad, pad + lh * i), line, fill=(255, 255, 255), font=font)
    return np.array(im)


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 25) -> Path:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    return path


def save_grid_gif(
    frame_lists: list[list[np.ndarray]],
    path: str | Path,
    fps: int = 25,
    axis: int = 1,
) -> Path:
    """Concatenate several equal-length frame sequences side by side."""
    n = min(len(f) for f in frame_lists)
    frames = [np.concatenate([f[i] for f in frame_lists], axis=axis) for i in range(n)]
    return save_gif(frames, path, fps=fps)
