"""Shared figure style for the blog post (every scripts/fig_*.py imports this).

The post renders in two Quarto themes (cosmo = white, darkly = #222). Figures are
saved with a transparent background, so every ink and series colour is picked
from the mid-lightness band that keeps >= ~3:1 contrast on both surfaces (run
this file to print the table). Identity is never colour alone: lines also get
markers/linestyles and direct labels where there are few series.

Usage:
    import style as S
    fig, ax = S.subplots(width=S.FULL, ncols=2)
    S.band(ax, steps, Y, S.C["planner"], "planner")
    S.save(fig, "ladder_curves")    # -> figures/ladder_curves.{svg,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
BLOG = HERE.parent
WS = BLOG.parent
FIG_DIR = BLOG / "figures"
VID_DIR = BLOG / "videos"
OUT = WS / "comfree_tdmpc" / "outputs"      # comfree_tdmpc run outputs (read-only)

# ---------------------------------------------------------------- colours
# Mid-band steps of a CVD-validated 8-hue categorical palette (blue, orange,
# aqua, yellow, magenta, green, violet, red), readable on white and on #222.
INK = "#7f7f7f"          # text, ticks, spines: 3.9:1 on white, 4.3:1 on #222
GRID = "#7f7f7f"         # drawn at low alpha
PENDING = "#8c8c8c"

C = {
    # fixed roles, used identically in every figure
    "planner": "#3987e5",    # MPPI through ComFree
    "policy": "#d95926",     # policy network alone
    "baseline": "#9085e9",   # TD-MPC2 / TD-M(PC)² (pearl), learned world model
    "pure_rl": "#c98500",    # SAC-style actor-critic, no planner
    "dagger": "#c98500",     # DAgger beta=0.3 arm (A2 only; shares the pure-RL hue, never shown together)
    # ablation ladder: A-C (no terminal Q) in greys, D-F in distinct hues
    "A": "#a6a6a6", "B": "#959595", "C": "#787878",
    "D": "#199e70", "E": "#d55181", "F": "#e66767",
    # SysID parameter groups
    "joint": "#959595", "actuator": "#9085e9", "contact": "#c98500",
    "inertial": "#e66767", "comfree": "#199e70",
    # diagram: learned vs given/analytic
    "learned": "#d95926", "given": "#3987e5",
}
MARKER = {"A": "o", "B": "s", "C": "D", "D": "^", "E": "v", "F": "P",
          "planner": "o", "policy": "s", "baseline": "D", "pure_rl": "x"}
RUNG_NAME = {"A": "A base", "B": "B +distill", "C": "C +SysID", "D": "D +terminal Q",
             "E": "E +bc actor", "F": "F +mean target"}
RUNG_DIR = {"A": "a_base", "B": "b_distill", "C": "c_sysid", "D": "d_termq", "E": "e_bc", "F": "f_meantarget"}

# ---------------------------------------------------------------- sizes
# Blog column ~750 px. Figures are drawn at 100 px/inch of column, so a 7.5 in
# figure fills the column and a 10 pt label shows at ~14 px. PNGs are 200 dpi (2x).
FULL = 7.5
HALF = 3.7
FS = {"base": 10, "small": 8.5, "title": 10.5, "annot": 8.5}
DPI = 200

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS["base"],
    "axes.titlesize": FS["title"],
    "axes.labelsize": FS["base"],
    "xtick.labelsize": FS["small"],
    "ytick.labelsize": FS["small"],
    "legend.fontsize": FS["small"],
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
    "axes.facecolor": "none", "figure.facecolor": "none", "savefig.facecolor": "none",
    "savefig.transparent": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 2.0, "lines.markersize": 5,
    "legend.frameon": False,
    "svg.fonttype": "path",   # glyphs as paths: identical in every browser
    "hatch.color": PENDING, "hatch.linewidth": 0.8,
})


def subplots(width=FULL, height=None, nrows=1, ncols=1, **kw):
    height = height or (3.0 if width == FULL else 2.8) * nrows
    return plt.subplots(nrows, ncols, figsize=(width, height), layout="constrained", **kw)


def save(fig, name: str, out_dir: Path | None = None):
    """figures/<name>.svg and .png (200 dpi, tight, transparent)."""
    d = Path(out_dir) if out_dir else FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(d / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    print(f"[fig] {d / name}.svg/.png")


def mean_std(Y):
    """Y: (reps, T) with NaN for missing -> mean, sample std (ddof=1, 0 if n=1), n per column."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    n = np.isfinite(Y).sum(0)
    m = np.nanmean(Y, 0)
    s = np.where(n > 1, np.nanstd(Y, 0, ddof=1) if Y.shape[0] > 1 else 0.0, 0.0)
    return m, s, n


def band(ax, x, Y, color, label=None, marker=None, ls="-", alpha=0.18, lw=2.0, ms=4.5, zorder=3, **kw):
    """Mean line ± sample-std band over reps. Y: (reps, len(x)). Returns the mean line."""
    m, s, _ = mean_std(Y)
    ax.fill_between(x, m - s, m + s, color=color, alpha=alpha, lw=0, zorder=zorder - 1)
    (ln,) = ax.plot(x, m, color=color, ls=ls, lw=lw, marker=marker, ms=ms, label=label, zorder=zorder,
                    markeredgecolor="none" if marker not in ("x", "+") else color, **kw)
    return ln


def pending(ax, text="TD-MPC2 baseline: run in progress", xy=(0.5, 0.5), rect=None, fontsize=None):
    """Grey PENDING annotation for slots that later scripts fill in.

    rect=(x0, y0, w, h) in axes coords also draws a hatched box where the data will go.
    """
    if rect is not None:
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle(rect[:2], rect[2], rect[3], transform=ax.transAxes, fill=False,
                               hatch="///", edgecolor=PENDING, lw=0.8, ls="--", alpha=0.7, zorder=1))
    ax.text(*xy, f"PENDING\n{text}", transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize or FS["annot"], color=PENDING, style="italic", zorder=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="none", ec=PENDING, lw=0.8, ls="--"))


def direct_label(ax, x, y, text, color, dx=4, dy=0, ha="left", **kw):
    """Label at the end of a line, in ink colour with a coloured marker-free offset."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                fontsize=FS["annot"], color=INK, **kw)


def kfmt(ax, axis="x"):
    """Env-step axis as 0, 50k, 100k, ..., 1M."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: "0" if v == 0 else (f"{v/1e6:g}M" if v >= 1e6 else f"{v/1e3:g}k"))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


# ---------------------------------------------------------------- self-test
def _lum(h):
    c = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


if __name__ == "__main__":
    print(f"{'role':10s} {'hex':8s} {'vs white':>9s} {'vs #222':>8s}")
    for k, v in {"ink": INK, **C}.items():
        print(f"{k:10s} {v:8s} {contrast(v, '#ffffff'):9.2f} {contrast(v, '#222222'):8.2f}")
    sys.exit(0)
