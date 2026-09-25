#!/usr/bin/env python
"""V1 videos/hero.mp4 (+ figures/hero_poster.png) and V2 videos/planner_vs_policy.mp4 from run GIFs.

V1: hero ablation_300k_rep2/e_bc, s0300000_mpc.gif | s0300000_pi.gif side by side (separate episodes),
    with a label strip on top. V2: comfree_tdmpc/outputs/report/gifs/d_termq_side_by_side.gif as-is.
H.264, yuv420p, even dimensions, no audio, +faststart, 25 fps (the GIFs' 40 ms frames).
Uses the ffmpeg binary shipped with imageio-ffmpeg (no system ffmpeg needed).
"""
from pathlib import Path

import imageio.v2 as iio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import runs as R
import style as S

FPS = 25
FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
HERO = S.OUT / R.RESULTS["hero"]["hero"] / "videos"
V2_SRC = S.OUT / "report" / "gifs" / "d_termq_side_by_side.gif"


def frames(p):
    return [np.asarray(f)[..., :3] for f in iio.get_reader(p)]


def label_strip(width, labels, h=30):
    """Dark strip with one centred label per equal-width column."""
    im = Image.new("RGB", (width, h), (34, 34, 34))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(str(FONT), 16)
    cw = width // len(labels)
    for i, t in enumerate(labels):
        tw = d.textlength(t, font=font)
        d.text((i * cw + (cw - tw) / 2, (h - 18) / 2), t, fill=(235, 235, 235), font=font)
    return np.asarray(im)


def even(a):
    h, w = a.shape[:2]
    return a[: h - h % 2, : w - w % 2]


def write_mp4(path, fr):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = iio.get_writer(path, fps=FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                       ffmpeg_params=["-movflags", "+faststart", "-crf", "20", "-preset", "slow", "-an"])
    for f in fr:
        w.append_data(even(f))
    w.close()
    print(f"[video] {path}  {path.stat().st_size / 1e6:.2f} MB, {len(fr)} frames, {len(fr) / FPS:.1f} s")


# V1 hero
mpc, pi = frames(HERO / "s0300000_mpc.gif"), frames(HERO / "s0300000_pi.gif")
n = min(len(mpc), len(pi))
body = [np.concatenate([mpc[i], np.full((mpc[i].shape[0], 4, 3), 34, np.uint8), pi[i]], 1) for i in range(n)]
strip = label_strip(body[0].shape[1], ["Planner (MPPI through ComFree)", "Policy alone"])
v1 = [np.concatenate([strip, b], 0) for b in body]
write_mp4(S.VID_DIR / "hero.mp4", v1)
S.FIG_DIR.mkdir(exist_ok=True)
iio.imwrite(S.FIG_DIR / "hero_poster.png", even(v1[60]))
print(f"[poster] {S.FIG_DIR / 'hero_poster.png'} (frame 60)")

# V2 planner vs policy, rung D
write_mp4(S.VID_DIR / "planner_vs_policy.mp4", frames(V2_SRC))
