"""Every figure and table in the report, from the json/csv the runs left behind.

Safe to run while the ablation is still going: anything missing is skipped and
whatever exists is plotted, so the figures can be checked early instead of once
at the end.

    python scripts/report/make_ablation_figures.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical palette, slots 1-6.  Used in this fixed order and never cycled;
# every figure also has a table in the report, which is what the low-contrast
# slots require.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b9b8b2"
GRID = "#e6e5e0"

# Wide figures are scaled down to the page width in the PDF (a 12.5-inch figure
# becomes 6.9 inches, roughly 55%).  Type is drawn oversized here so that it
# lands at a readable size after that reduction rather than at 5pt.
FS = 1.25

RUNS = [
    ("a_base", "A  base", C[0]),
    ("b_distill", "B  + distill", C[1]),
    ("c_sysid", "C  + SysID", C[2]),
    ("d_termq", "D  + terminal Q", C[3]),
    ("e_bc", "E  + BC policy", C[4]),
    ("f_meantarget", "F  + single-mode target", C[5]),
]


def style(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9 * FS, length=3, width=0.8)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10 * FS)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10 * FS)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5 * FS, loc="left", pad=8,
                     weight="bold")


def legend(ax, **kw):
    kw["fontsize"] = kw.get("fontsize", 9) * FS
    return ax.legend(frameon=False, labelcolor=INK2, **kw)


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


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def smooth(x: np.ndarray, k: int = 7) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < k or k < 2:
        return x
    pad = np.concatenate([np.full(k // 2, x[0]), x, np.full(k // 2, x[-1])])
    return np.convolve(pad, np.ones(k) / k, mode="valid")[: len(x)]


def curve(ax, x, y, label, color, k=7, lw=2.0):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return
    ax.plot(x[m], y[m], color=color, alpha=0.18, lw=1.0, zorder=2)
    ax.plot(x[m], smooth(y[m], k), color=color, lw=lw, label=label, zorder=3)


def save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


# figure 1
def fig_throughput(paths: dict, out: Path) -> None:
    thr = read_json(paths["throughput"])
    vec = read_json(paths["verify"])
    if not thr and not vec:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.0), constrained_layout=True)

    if thr:
        s = thr["sim"]
        n = np.array([r["nworld"] for r in s])
        ms = np.array([r["ctrl_step_ms"] for r in s])
        ax = axes[0]
        ideal = ms[0] * n / n[0]
        ax.plot(n, ideal, color=MUTED, lw=1.6, ls="--", zorder=2,
                label="if width cost linearly")
        ax.plot(n, ms, color=C[0], lw=2.0, marker="o", ms=6, zorder=3,
                label="measured")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(1, max(ideal) * 3.0)
        style(ax, "worlds simulated together", "milliseconds per control step",
              "1  Width is nearly free")
        ax.annotate(f"{n[0]} world\n{ms[0]:.1f} ms", (n[0], ms[0]),
                    textcoords="offset points", xytext=(10, 2),
                    fontsize=8.5 * FS, color=INK2)
        ax.annotate(f"{n[-1]} worlds\n{ms[-1]:.1f} ms", (n[-1], ms[-1]),
                    textcoords="offset points", xytext=(-8, 14),
                    ha="right", fontsize=8.5 * FS, color=INK2)
        legend(ax, loc="upper left")

    if vec and vec.get("throughput"):
        t = vec["throughput"]
        e = np.array([r["num_envs"] for r in t])
        sps = np.array([r["env_steps_per_s"] for r in t])
        el = np.array([r["elite_transitions_per_s"] for r in t])

        ax = axes[1]
        ax.plot(e, sps, color=C[0], lw=2.0, marker="o", ms=7, zorder=3)
        style(ax, "environments planned together", "environment steps per second",
              "2  Data rate, same GPU")
        for xi, yi in zip(e, sps):
            ax.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8.5 * FS, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(e)
        ax.set_xticklabels([str(int(v)) for v in e])
        ax.set_ylim(0, max(sps) * 1.35)
        ax.annotate(f"{sps[-1] / sps[0]:.1f}x more data\nfrom the same device",
                    (e[-1], sps[-1]), textcoords="offset points", xytext=(-10, -46),
                    ha="right", fontsize=10 * FS, color=C[0], weight="bold")

        ax = axes[2]
        ax.plot(e, el, color=C[2], lw=2.0, marker="o", ms=7, zorder=3,
                label="MPPI elite transitions")
        ax.plot(e, sps, color=C[0], lw=2.0, marker="o", ms=7, zorder=3,
                label="executed transitions")
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks(e)
        ax.set_xticklabels([str(int(v)) for v in e])
        ax.set_ylim(min(sps) * 0.4, max(el) * 4)
        style(ax, "environments planned together", "training transitions per second",
              "3  What the learner sees")
        legend(ax, loc="upper left", ncol=1)

    fig.suptitle("Throughput: the planner is deep, not wide -- so fill the width",
                 fontsize=13.5 * FS, color=INK, weight="bold", x=0.01, ha="left")
    save(fig, out, "fig1_throughput.png")


# figure 2
def fig_planner_tuning(paths: dict, out: Path) -> None:
    rows = read_json(paths["tune"])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    x = np.array([r["ms_per_env_step"] for r in rows])
    y = np.array([r["goals_per_1k_steps"] for r in rows])
    base = next((r for r in rows if r["config"] == "base"), rows[0])
    chosen = next((r for r in rows if r["config"] == "iterations=3"), None)

    # Labels are placed by hand: eight points in a small plane collide under any
    # uniform offset, and a legend cannot carry eight one-off configurations.
    offsets = {
        "base": (12, 0), "iterations=3": (-14, 10), "iterations=2": (11, 3),
        "iterations=1": (11, 3), "num_samples=128": (-11, 7),
        "num_samples=512": (-10, 11), "horizon=4": (11, -13), "horizon=6": (11, -13),
        "horizon=12": (-10, 11),
    }
    for r, xi, yi in zip(rows, x, y):
        is_base = r["config"] == "base"
        is_chosen = chosen is not None and r["config"] == chosen["config"]
        col = C[1] if is_base else (C[2] if is_chosen else C[0])
        ax.scatter([xi], [yi], s=170 if (is_base or is_chosen) else 70,
                   color=col, zorder=4, edgecolor="white", linewidth=1.5)
        dx, dy = offsets.get(r["config"], (8, 8))
        ax.annotate(r["config"], (xi, yi), textcoords="offset points",
                    xytext=(dx, dy), ha="right" if dx < 0 else "left",
                    fontsize=9 * FS, color=INK2, zorder=5)
    style(ax, "cost:  milliseconds per environment step",
          "quality:  goals solved per 1000 steps",
          "Planner configurations: quality against cost")
    ax.set_xlim(0, max(x) * 1.25)
    ax.set_ylim(0, max(y) * 1.25)
    if chosen:
        q = 100 * chosen["goals_per_1k_steps"] / base["goals_per_1k_steps"]
        c = 100 * chosen["ms_per_env_step"] / base["ms_per_env_step"]
        ax.annotate(f"chosen for the ablation\n{q:.0f}% of the quality, "
                    f"{c:.0f}% of the cost",
                    (chosen["ms_per_env_step"], chosen["goals_per_1k_steps"]),
                    textcoords="offset points", xytext=(110, -125), fontsize=9.5,
                    color=C[2], weight="bold", ha="center",
                    arrowprops=dict(arrowstyle="->", color=C[2], lw=1.4,
                                    connectionstyle="arc3,rad=-0.25"))
        ax.annotate("better", (0.055, 0.93), xycoords="axes fraction",
                    fontsize=9 * FS, color=MUTED, ha="center")
        ax.annotate("", (0.03, 0.97), (0.03, 0.80), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    save(fig, out, "fig2_planner_tuning.png")


# figure 3
def fig_learning_curves(runs: dict[str, Path], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.0))
    panels = [
        ("episode_reward", "reward per episode", ""),
        ("goals_reached", "goals solved per episode", ""),
        ("dropped", "episodes ending in a drop", ""),
    ]
    any_data = False
    for ax, (key, ylabel, title) in zip(axes, panels):
        for name, label, col in RUNS:
            d = read_csv(runs.get(name, Path("x")) / "train.csv")
            if not d or key not in d:
                continue
            any_data = True
            curve(ax, d["step"], d[key], label, col)
        style(ax, "environment steps", ylabel, title)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    if not any_data:
        plt.close(fig)
        return
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, labelcolor=INK2, fontsize=9 * FS,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Training behaviour of the closed loop (the planner is driving)",
                 fontsize=13 * FS, color=INK, weight="bold", x=0.005, ha="left", y=1.05)
    save(fig, out, "fig3_learning_curves.png")


# figure 4
def fig_policy_vs_planner(runs: dict[str, Path], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)
    any_data = False
    for name, label, col in RUNS:
        d = read_csv(runs.get(name, Path("x")) / "eval.csv")
        if not d:
            continue
        if "pi/return" in d:
            any_data = True
            curve(axes[0], d["step"], d["pi/return"], label, col, k=3)
        if "mpc/return" in d:
            m = np.isfinite(d["mpc/return"])
            if m.sum():
                axes[1].plot(d["step"][m], d["mpc/return"][m], color=col, lw=2.0,
                             marker="o", ms=6, label=label, zorder=3)
    if not any_data:
        plt.close(fig)
        return
    style(axes[0], "environment steps", "reward per episode", "1  Policy alone")
    style(axes[1], "environment steps", "reward per episode", "2  MPPI planner")
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    xlo = min(a.get_xlim()[0] for a in axes)
    xhi = max(a.get_xlim()[1] for a in axes)
    for a in axes:
        # One shared scale on both axes: the distance between the panels is the
        # whole point, and it is only readable if nothing else differs.
        a.set_ylim(lo, hi)
        a.set_xlim(xlo, xhi)
        a.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    legend(axes[0], loc="upper left")
    save(fig, out, "fig4_policy_vs_planner.png")


# figure 5
def fig_sysid(runs: dict[str, Path], out: Path) -> None:
    have = [(n, l, c) for n, l, c in RUNS
            if (runs.get(n, Path("x")) / "sysid.csv").exists()]
    if not have:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.0))

    for name, label, col in RUNS:
        d = read_csv(runs.get(name, Path("x")) / "train.csv")
        if d and "param_err_mean" in d:
            curve(axes[0], d["step"], d["param_err_mean"], label, col, k=3)
    style(axes[0], "environment steps", "mean |theta_hat - theta_true| (normalised)",
          "1  How wrong the planner's physics is")
    legend(axes[0], loc="best")

    for name, label, col in have:
        d = read_csv(runs[name] / "sysid.csv")
        if not d:
            continue
        axes[1].plot(d["step"], d["gap_closed"], color=col, lw=2.0,
                     marker="o", ms=5, label=label, zorder=3)
    axes[1].axhline(0, color=MUTED, lw=1.0, ls="--", zorder=2)
    style(axes[1], "environment steps", "percent of the loss gap closed",
          "2  Each identification call")
    for a in (axes[0], axes[1]):
        a.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))

    name = have[0][0]
    d = read_csv(runs[name] / "sysid.csv")
    groups = sorted(k for k in d if k.startswith("err/"))
    if groups:
        first = np.array([d[g][0] for g in groups])
        last = np.array([d[g][-1] for g in groups])
        y = np.arange(len(groups))
        axes[2].barh(y - 0.19, first, height=0.36, color=MUTED, zorder=3,
                     label="first call")
        axes[2].barh(y + 0.19, last, height=0.36, color=C[2], zorder=3,
                     label="last call")
        axes[2].set_yticks(y)
        axes[2].set_yticklabels([g.split("/")[1] for g in groups], fontsize=9)
        style(axes[2], "mean parameter error (normalised)", "",
              f"3  Which parameters moved ({have[0][1].strip()})")
        legend(axes[2], loc="lower right")
    fig.suptitle("System identification: 71 unknown parameters, fitted from replayed chunks",
                 fontsize=13 * FS, color=INK, weight="bold", x=0.005, ha="left", y=1.05)
    save(fig, out, "fig5_sysid.png")


# figure 6
def fig_final_bars(runs: dict[str, Path], out: Path) -> dict:
    table = {}
    for name, label, col in RUNS:
        ev = read_csv(runs.get(name, Path("x")) / "eval.csv")
        tr = read_csv(runs.get(name, Path("x")) / "train.csv")
        if not ev:
            continue
        row = {"label": label, "color": col}
        # The tail mean, not the single last evaluation: one evaluation is 16
        # episodes of a task dominated by whether the cube survives, and a lone
        # final number moves by more than the effects being compared.
        for k, win in (("pi/return", 6), ("pi/goals_reached", 6), ("pi/dropped", 6),
                       ("mpc/return", 3), ("mpc/goals_reached", 3), ("mpc/dropped", 3)):
            if k in ev:
                v = ev[k][np.isfinite(ev[k])]
                row[k] = float(v[-win:].mean()) if len(v) else np.nan
                row[k + "/std"] = float(v[-win:].std()) if len(v) else np.nan
        if tr:
            row["steps"] = float(tr["step"][-1])
            row["env_steps_per_s"] = float(tr["env_steps_per_s"][-1])
            row["updates"] = float(tr["updates"][-1])
            if "param_err_mean" in tr:
                row["param_err_final"] = float(tr["param_err_mean"][-1])
                row["param_err_init"] = float(tr["param_err_mean"][0])
        table[name] = row
    if not table:
        return {}

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), constrained_layout=True)
    names = list(table)
    y = np.arange(len(names))
    # The two quantities are named on the x axes rather than in panel titles:
    # a left-aligned suptitle and a panel title land on the same line.
    for ax, (kp, km, title) in zip(axes, [
        ("pi/return", "mpc/return", "reward per episode"),
        ("pi/goals_reached", "mpc/goals_reached", "goals solved per episode"),
    ]):
        pi = [table[n].get(kp, np.nan) for n in names]
        mp = [table[n].get(km, np.nan) for n in names]
        pie = [table[n].get(kp + "/std", np.nan) for n in names]
        mpe = [table[n].get(km + "/std", np.nan) for n in names]
        ekw = dict(ecolor=INK2, elinewidth=1.2, capsize=3)
        ax.barh(y + 0.19, mp, height=0.36, color=C[0], zorder=3, label="MPPI planner",
                xerr=mpe, error_kw=ekw)
        ax.barh(y - 0.19, pi, height=0.36, color=C[1], zorder=3, label="learned policy",
                xerr=pie, error_kw=ekw)
        ax.set_yticks(y)
        ax.set_yticklabels([table[n]["label"] for n in names], fontsize=9.5 * FS)
        ax.invert_yaxis()
        style(ax, title, "", "")
        ax.axvline(0, color=MUTED, lw=1.0)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, labelcolor=INK2, fontsize=9 * FS,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Final performance of every rung on the ladder",
                 fontsize=13 * FS, color=INK, weight="bold", x=0.005, ha="left", y=1.02)
    save(fig, out, "fig6_final_bars.png")
    return table


# figure 7
def fig_diagnosis(runs: dict[str, Path], out: Path) -> None:
    """Why the policy does not match the planner, in one picture."""
    data = {}
    for name, label, col in RUNS:
        p = runs.get(name, Path("x")) / "diagnosis.json"
        if p.exists():
            data[name] = (label, col, json.loads(p.read_text()))
    if not data:
        return

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.0), constrained_layout=True)
    names = list(data)
    y = np.arange(len(names))
    short = {"a_base": "A base", "b_distill": "B distill", "c_sysid": "C SysID",
             "d_termq": "D term Q", "e_bc": "E BC", "f_meantarget": "F mean"}
    ylab = [short.get(n, data[n][0]) for n in names]

    ax = axes[0]
    on_plan = [data[n][2].get("imitation_error/planner_states", np.nan) for n in names]
    on_pi = [data[n][2].get("imitation_error/policy_states", np.nan) for n in names]
    floor = [data[n][2].get("elite_spread/mean_abs_dev", np.nan) for n in names]
    ax.barh(y - 0.22, on_plan, height=0.4, color=C[0], zorder=3,
            label="on the planner's states")
    ax.barh(y + 0.22, on_pi, height=0.4, color=C[1], zorder=3,
            label="on the policy's own states")
    for i, f in enumerate(floor):
        ax.plot([f, f], [i - 0.45, i + 0.45], color=INK, lw=2.4, zorder=5,
                label="floor: spread inside the elite set" if i == 0 else None)
    ax.set_yticks(y)
    ax.set_yticklabels(ylab, fontsize=9.5 * FS)
    ax.invert_yaxis()
    style(ax, "mean absolute action error", "", "1  Can it copy the planner?")
    ax.set_xlim(0, np.nanmax(on_pi + on_plan + floor) * 1.15)
    legend(ax, loc="lower center", fontsize=7.4, bbox_to_anchor=(0.5, -0.52))

    # The ratio, not the two errors side by side: copying accuracy improves all
    # the way down the ladder while performance peaks in the middle, and this is
    # the panel that shows what breaks the link.
    ax = axes[1]
    shift = [(data[n][2].get("imitation_error/policy_states", np.nan)
              / data[n][2].get("imitation_error/planner_states", np.nan))
             for n in names]
    ax.barh(y, shift, height=0.55, color=[data[n][1] for n in names], zorder=3)
    for i, v in enumerate(shift):
        if np.isfinite(v):
            ax.annotate(f"{v:.2f}x", (v, i), textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=9 * FS, color=INK2)
    ax.axvline(1.0, color=MUTED, lw=1.4, ls="--", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(ylab, fontsize=9.5 * FS)
    ax.invert_yaxis()
    style(ax, "own states / planner states", "", "2  How far it drifts")
    ax.set_xlim(0, np.nanmax(shift) * 1.3)

    ax = axes[2]
    q = [data[n][2].get("value/q_mean", np.nan) for n in names]
    mc = [data[n][2].get("value/mc_return_mean", np.nan) for n in names]
    ax.barh(y - 0.22, mc, height=0.4, color=C[0], zorder=3,
            label="return that actually followed")
    ax.barh(y + 0.22, q, height=0.4, color=C[1], zorder=3, label="what Q predicted")
    ax.set_yticks(y)
    ax.set_yticklabels(ylab, fontsize=9.5 * FS)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=1.0)
    style(ax, "value over the next 20 steps", "", "3  Is Q honest?")
    lo = min(0.0, np.nanmin(q + mc) * 1.15)
    ax.set_xlim(lo, np.nanmax(q + mc) * 1.15)
    legend(ax, loc="lower center", fontsize=7.4, bbox_to_anchor=(0.5, -0.52))

    fig.suptitle("Diagnosing the distillation gap",
                 fontsize=13 * FS, color=INK, weight="bold", x=0.005, ha="left")
    fig.get_layout_engine().set(w_pad=0.10, wspace=0.09)
    save(fig, out, "fig7_diagnosis.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", type=str, default="outputs/ablation")
    ap.add_argument("--out", type=str, default="outputs/report/figures")
    args = ap.parse_args()

    root = Path(args.ablation)
    runs = {n: root / n for n, _, _ in RUNS}
    out = Path(args.out)
    paths = {
        "throughput": Path("outputs/06_throughput/throughput.json"),
        "verify": Path("outputs/07_verify/vec.json"),
        "tune": Path("outputs/08_tune/planner.json"),
    }

    print("figures:")
    fig_throughput(paths, out)
    fig_planner_tuning(paths, out)
    fig_learning_curves(runs, out)
    fig_policy_vs_planner(runs, out)
    fig_sysid(runs, out)
    fig_diagnosis(runs, out)
    table = fig_final_bars(runs, out)
    if table:
        (out / "summary.json").write_text(json.dumps(table, indent=2, default=str))
        print(f"  wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
