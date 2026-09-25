"""Turn the finished runs in outputs/runs into the report figures and gifs.

    python scripts/report/make_figures.py

Every figure compares a matched pair that differs in exactly one thing, and
every gif states in the frame which controller is driving.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from comfree_tdmpc.logger import new_figure
from comfree_tdmpc.sim.params import PARAM_NAMES

# Return of holding the scene's initial posture for a full episode, measured by
# scripts/report/policy_baselines.py.  Random actions score -403.
HOLD_BASELINE = -114.4

PAIRS = {
    "sysid": (
        ("sysid_on", "SysID on", "tab:blue"),
        ("sysid_off", "SysID off", "tab:red"),
        "Wrong initial physics",
    ),
    "terminal": (
        ("termQ_on", "with learned terminal Q", "tab:green"),
        ("termQ_off", "no terminal value", "tab:gray"),
        "Short-horizon planner",
    ),
    "distill": (
        ("distill_on", "trained on MPPI elite rollouts", "tab:purple"),
        ("distill_off", "trained on executed actions only", "tab:brown"),
        "Policy and value training data",
    ),
}


def read_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[k] = np.array(vals)
    return out


def smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < k or k < 2:
        return x
    pad = np.concatenate([np.full(k // 2, x[0]), x, np.full(k // 2, x[-1])])
    return np.convolve(pad, np.ones(k) / k, mode="valid")[: len(x)]


def curve(ax, x, y, label, color, k=5):
    ax.plot(x, y, color=color, alpha=0.2, lw=1)
    ax.plot(x, smooth(y, k), color=color, lw=2, label=label)


def fig_pair(runs: dict[str, Path], out: Path, key: str) -> None:
    (a, la, ca), (b, lb, cb), title = PAIRS[key]
    ta, tb = read_csv(runs.get(a, Path("x")) / "train.csv"), read_csv(runs.get(b, Path("x")) / "train.csv")
    if not ta and not tb:
        return
    ea, eb = read_csv(runs.get(a, Path("x")) / "eval.csv"), read_csv(runs.get(b, Path("x")) / "eval.csv")

    n = 4 if key == "sysid" else 3
    fig, axes = new_figure(1, n, figsize=(4.0 * n, 3.7))

    for t, label, color in ((ta, la, ca), (tb, lb, cb)):
        if t:
            curve(axes[0], t["episode"], t["episode_reward"], label, color)
            curve(axes[1], t["episode"], t["goals_reached"], label, color)
    axes[0].set_ylabel("episode return (planner)")
    axes[0].set_title(f"{title}: return")
    axes[1].set_ylabel("goals reached / episode")
    axes[1].set_title("Task progress")

    # The policy is evaluated on its own, with no planner in the loop.
    for e, label, color in ((ea, la, ca), (eb, lb, cb)):
        if e and "pi/return" in e:
            axes[2].plot(e["episode"], e["pi/return"], "-o", ms=4, color=color,
                         label=f"{label}: policy alone")
            axes[2].plot(e["episode"], e["mpc/return"], "--s", ms=3, color=color, alpha=0.5,
                         label=f"{label}: planner")
    # Measured floor: holding the initial posture never drops the cube and never
    # solves anything.  Without it a negative policy return is unreadable.
    axes[2].axhline(HOLD_BASELINE, color="k", ls=":", lw=1.2,
                    label="do nothing (hold posture)")
    axes[2].set_ylabel("evaluation return")
    axes[2].set_title("Policy alone vs. its planner")

    if key == "sysid":
        for t, label, color in ((ta, la, ca), (tb, lb, cb)):
            if t:
                axes[3].plot(t["episode"], t["param_err_mean"], color=color, lw=2, label=label)
        axes[3].set_yscale("log")
        axes[3].set_ylabel(r"mean $|\log\hat\theta-\log\theta^*|$")
        axes[3].set_title("Physical parameter error")

    for ax in axes:
        ax.set_xlabel("training episode")
        ax.legend(fontsize=6)
    fig.tight_layout()
    path = out / f"fig_{key}.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def fig_sysid_params(runs: dict[str, Path], out: Path, true_params: np.ndarray) -> None:
    run = runs.get("sysid_on")
    if run is None:
        return
    sy = read_csv(run / "sysid.csv")
    if not sy:
        return
    fig, axes = new_figure(1, 2, figsize=(11, 3.6))
    for i, n in enumerate(PARAM_NAMES):
        if f"param/{n}" in sy:
            axes[0].plot(sy["episode"], sy[f"param/{n}"] / true_params[i], "-o", ms=3, label=n)
    axes[0].axhline(1.0, color="k", ls="--", lw=1, label="truth")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("estimate / truth")
    axes[0].set_title("Online SysID during training")
    axes[1].semilogy(sy["episode"], sy["val_loss"], "-o", ms=3, label="held-out prediction loss")
    axes[1].semilogy(sy["episode"], sy["param_err_mean"], "-s", ms=3,
                     label=r"mean $|\log\hat\theta-\log\theta^*|$")
    axes[1].set_title("SysID objective and parameter error")
    for ax in axes:
        ax.set_xlabel("training episode")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_sysid_params.png", dpi=140)
    print(f"wrote {out}/fig_sysid_params.png")


def fig_policy_gap(runs: dict[str, Path], out: Path) -> None:
    """How close does the distilled policy get to the planner that taught it?"""
    fig, axes = new_figure(1, 2, figsize=(11, 3.7))
    any_data = False
    for name, run in sorted(runs.items()):
        e = read_csv(run / "eval.csv")
        if not e or "pi/return" not in e:
            continue
        any_data = True
        axes[0].plot(e["episode"], e["pi/return"], "-o", ms=3, label=f"{name}: policy")
        axes[0].plot(e["episode"], e["mpc/return"], "--", lw=1, alpha=0.45,
                     color=axes[0].lines[-1].get_color(), label=f"{name}: planner")
        denom = np.where(np.abs(e["mpc/return"]) < 1e-6, np.nan, e["mpc/return"])
        axes[1].plot(e["episode"], 100 * e["pi/return"] / denom, "-o", ms=3, label=name)
    if not any_data:
        return
    axes[1].axhline(100, color="k", ls="--", lw=1, label="parity with the planner")
    axes[0].axhline(HOLD_BASELINE, color="k", ls=":", lw=1.2, label="do nothing")
    axes[0].set_ylabel("evaluation return")
    axes[0].set_title("Policy alone vs. planner, per run")
    axes[1].set_ylabel("policy return as % of planner return")
    axes[1].set_ylim(-150, 150)
    axes[1].set_title("Distillation gap")
    for ax in axes:
        ax.set_xlabel("training episode")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig_policy_gap.png", dpi=140)
    print(f"wrote {out}/fig_policy_gap.png")


def fig_losses(runs: dict[str, Path], out: Path) -> None:
    fig, axes = new_figure(1, 3, figsize=(15, 3.5))
    for name, run in sorted(runs.items()):
        t = read_csv(run / "train.csv")
        if not t or "value_loss" not in t:
            continue
        m = ~np.isnan(t["value_loss"])
        if m.sum() == 0:
            continue
        axes[0].plot(t["episode"][m], smooth(t["value_loss"][m]), label=name)
        if "reward_loss" in t:
            axes[1].plot(t["episode"][m], smooth(t["reward_loss"][m]), label=name)
        if "pi_loss" in t:
            axes[2].plot(t["episode"][m], smooth(t["pi_loss"][m]), label=name)
    for ax, title in zip(axes, ["Q (two-hot) loss", "reward-model loss", "policy loss"]):
        ax.set_xlabel("training episode")
        ax.set_title(title)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig_losses.png", dpi=140)
    print(f"wrote {out}/fig_losses.png")


def _load(path: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    return [np.array(f)[..., :3] for f in imageio.mimread(path, memtest=False)]


def _write(frames: list[np.ndarray], path: Path, fps: int = 30) -> None:
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"wrote {path} ({len(frames)} frames)")


def _pad(a: list[np.ndarray], n: int) -> list[np.ndarray]:
    return a + [a[-1]] * (n - len(a))


def progress_gif(run: Path, out: Path, name: str, kind: str) -> None:
    vids = sorted((run / "videos").glob(f"*_{kind}.gif"))
    if not vids:
        return
    _write([f for v in vids for f in _load(v)], out / f"{name}_{kind}_progress.gif")


def planner_vs_policy_gif(run: Path, out: Path, name: str) -> None:
    stems = sorted({p.name[:-8] for p in (run / "videos").glob("*_mpc.gif")}
                   & {p.name[:-7] for p in (run / "videos").glob("*_pi.gif")})
    if not stems:
        return
    frames = []
    for stem in stems:
        a, b = _load(run / "videos" / f"{stem}_mpc.gif"), _load(run / "videos" / f"{stem}_pi.gif")
        n = max(len(a), len(b))
        frames += [np.concatenate(p, axis=1) for p in zip(_pad(a, n), _pad(b, n))]
    _write(frames, out / f"{name}_planner_vs_policy.gif")


def paired_gif(runs: dict[str, Path], out: Path, key: str, kind: str = "mpc") -> None:
    (a, _, _), (b, _, _), _ = PAIRS[key]
    if a not in runs or b not in runs:
        return
    common = sorted({p.name for p in (runs[a] / "videos").glob(f"*_{kind}.gif")}
                    & {p.name for p in (runs[b] / "videos").glob(f"*_{kind}.gif")})
    if not common:
        return
    frames = []
    for nm in common:
        fa, fb = _load(runs[b] / "videos" / nm), _load(runs[a] / "videos" / nm)
        n = max(len(fa), len(fb))
        frames += [np.concatenate(p, axis=1) for p in zip(_pad(fa, n), _pad(fb, n))]
    _write(frames, out / f"compare_{key}_{kind}.gif")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default="outputs/runs")
    ap.add_argument("--out", type=str, default="outputs/report")
    ap.add_argument("--true_params", type=float, nargs=4, default=[1.0, 1.0, 0.1, 1e-3])
    ap.add_argument("--no_gifs", action="store_true")
    args = ap.parse_args()

    root, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {p.name: p for p in sorted(root.iterdir()) if p.is_dir() and (p / "train.csv").exists()}
    print("runs:", ", ".join(runs) or "(none)")
    if not runs:
        return

    for key in PAIRS:
        fig_pair(runs, out, key)
    fig_sysid_params(runs, out, np.array(args.true_params))
    fig_policy_gap(runs, out)
    fig_losses(runs, out)

    if not args.no_gifs:
        for name, run in runs.items():
            progress_gif(run, out, name, "mpc")
            progress_gif(run, out, name, "pi")
            planner_vs_policy_gif(run, out, name)
        for key in PAIRS:
            paired_gif(runs, out, key)

    lines = []
    for name, run in runs.items():
        t, e = read_csv(run / "train.csv"), read_csv(run / "eval.csv")
        if not t:
            continue
        last = slice(-10, None)
        line = (f"{name:<14} last10 return {np.nanmean(t['episode_reward'][last]):8.1f} | "
                f"goals {np.nanmean(t['goals_reached'][last]):4.2f} | "
                f"drop {np.nanmean(t['dropped'][last]):4.2f} | "
                f"theta_err {t['param_err_mean'][-1]:.3f}")
        if e and "pi/return" in e:
            line += (f" | final eval planner {e['mpc/return'][-1]:7.1f} "
                     f"policy {e['pi/return'][-1]:7.1f}")
        lines.append(line)
    summary = "\n".join(lines)
    print("\n" + summary)
    (out / "summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
