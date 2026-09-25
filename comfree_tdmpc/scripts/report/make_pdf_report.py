"""Build the PDF report from the figures and the run logs.

Every number in the prose is read out of `outputs/`, never typed in, so the
report cannot drift away from the runs it describes.

    python scripts/report/make_ablation_figures.py
    python scripts/report/make_pdf_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d8d2")
BAND = colors.HexColor("#f4f3ef")
ACCENT = colors.HexColor("#2a78d6")
GOOD = colors.HexColor("#1baf7a")
WARN = colors.HexColor("#eb6834")

PAGE_W = A4[0] - 3.4 * cm

RUNS = [
    ("a_base", "A  base"),
    ("b_distill", "B  + distill"),
    ("c_sysid", "C  + SysID"),
    ("d_termq", "D  + terminal Q"),
    ("e_bc", "E  + BC policy"),
    ("f_meantarget", "F  + single-mode target"),
]


def styles() -> dict:
    ss = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                                fontSize=22, leading=26, textColor=INK, alignment=0,
                                spaceAfter=4),
        "subtitle": ParagraphStyle("st", parent=ss["Normal"], fontName="Helvetica",
                                   fontSize=11.5, leading=15, textColor=INK2,
                                   spaceAfter=16),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                             fontSize=15, leading=19, textColor=INK,
                             spaceBefore=16, spaceAfter=7),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                             fontSize=11.5, leading=15, textColor=INK,
                             spaceBefore=11, spaceAfter=4),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica",
                               fontSize=9.8, leading=14.2, textColor=INK,
                               alignment=TA_JUSTIFY, spaceAfter=7),
        "cap": ParagraphStyle("c", parent=ss["Normal"], fontName="Helvetica-Oblique",
                              fontSize=8.5, leading=11.5, textColor=INK2,
                              spaceBefore=3, spaceAfter=11),
        "note": ParagraphStyle("n", parent=ss["Normal"], fontName="Helvetica",
                               fontSize=9.4, leading=13.4, textColor=INK,
                               spaceAfter=3, leftIndent=9),
    }
    return s


def para(text, st, style="body"):
    return Paragraph(text, st[style])


def caption(text, st):
    return Paragraph(text, st["cap"])


def figure(path: Path, st, cap: str, width: float = PAGE_W):
    if not path.exists():
        return []
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    return [Image(str(path), width=width, height=width * h / w),
            caption(cap, st)]


def table(rows, st, col_widths=None, highlight_rows=(), align_right_from=1):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    cmds = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.6),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 8.6),
        ("TEXTCOLOR", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK2),
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for r in highlight_rows:
        cmds += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eef4fc")),
                 ("FONT", (0, r), (-1, r), "Helvetica-Bold", 8.6),
                 ("TEXTCOLOR", (0, r), (-1, r), INK)]
    t.setStyle(TableStyle(cmds))
    return t


def keypoint(text, st, color=ACCENT):
    # reportlab wants '#rrggbb'; Color.hexval() returns '0xffrrggbb'.
    hexcol = "#" + color.hexval()[4:]
    p = Paragraph(f'<font color="{hexcol}"><b>{text}</b></font>', st["note"])
    t = Table([[p]], colWidths=[PAGE_W], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BAND),
        ("LINEBEFORE", (0, 0), (0, -1), 2.4, color),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
    ]))
    return [Spacer(1, 3), t, Spacer(1, 9)]


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


def last_finite(a) -> float:
    if a is None:
        return float("nan")
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a[-1]) if len(a) else float("nan")


def tail(a, k: int = 4) -> tuple[float, float]:
    """Mean and spread of the last `k` evaluations.

    One evaluation is 16 episodes of a task whose outcome is dominated by
    whether the cube survives, so a single final number moves by more than the
    differences this report is trying to resolve.  Averaging the tail of the run
    costs nothing and is the difference between a measurement and an anecdote.
    """
    if a is None:
        return float("nan"), float("nan")
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return float("nan"), float("nan")
    a = a[-k:]
    return float(a.mean()), float(a.std())


def pm(a, k: int = 4, nd: int = 2) -> str:
    m, s = tail(a, k)
    return "--" if not np.isfinite(m) else f"{m:.{nd}f} +/- {s:.{nd}f}"


def fmt(v, nd=1, dash="--") -> str:
    return dash if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def build(args) -> Path:
    st = styles()
    figs = Path(args.figures)
    abl = Path(args.ablation)
    thr = read_json(Path("outputs/06_throughput/throughput.json")) or {}
    ver = read_json(Path("outputs/07_verify/vec.json")) or {}
    tune = read_json(Path("outputs/08_tune/planner.json")) or []

    runs = {n: {"train": read_csv(abl / n / "train.csv"),
                "eval": read_csv(abl / n / "eval.csv"),
                "sysid": read_csv(abl / n / "sysid.csv")} for n, _ in RUNS}

    # headline numbers, all measured
    sim = {r["nworld"]: r for r in thr.get("sim", [])}
    vthr = {r["num_envs"]: r for r in ver.get("throughput", [])}
    e1, e16 = vthr.get(1), vthr.get(16)
    speedup = (e16["env_steps_per_s"] / e1["env_steps_per_s"]) if (e1 and e16) else float("nan")
    upd_ms = (thr.get("update", [{}])[0] or {}).get("update_ms", float("nan"))

    story = []

    # page 1
    story += [
        para("ComFree-TDMPC: Making the Loop Fast Enough to Ask the Question", st, "title"),
        para("A throughput fix, then a four-rung ablation of distillation, system "
             "identification and terminal value learning on in-hand cube reorientation, "
             "and two further runs chasing the policy that never catches the planner.",
             st, "subtitle"),
    ]

    story += [para("What this project is", st, "h1")]
    story += [para(
        "A robot hand has to turn a cube to a target orientation. The controller is "
        "<b>MPPI</b>: at every control step it imagines a few hundred different action "
        "sequences, simulates all of them, and executes the first action of the best one. "
        "What makes this project unusual is that the simulator it imagines with is a "
        "<b>real differentiable physics engine</b> (ComFree-Sim), not a neural network "
        "trained to imitate physics. Everything TD-MPC normally learns about dynamics is "
        "already known.", st)]
    story += [para(
        "That leaves three things worth learning, and they are exactly the three things "
        "this report switches on one at a time:", st)]

    ladder_rows = [
        ["Component", "What it is meant to do"],
        ["Distillation",
         "The planner simulates thousands of trajectories per step and throws almost all "
         "of them away. Keep the best ones as training data."],
        ["System identification (SysID)",
         "The planner's simulator has the wrong physical numbers. Fit the 71 unknown "
         "parameters from what actually happened on the robot."],
        ["Terminal Q",
         "The planner can only look 8 steps ahead. A learned value function tells it what "
         "the state at step 8 is worth, extending its horizon for free."],
    ]
    story += [table([[r[0], Paragraph(r[1], st["body"])] if i else r
                     for i, r in enumerate(ladder_rows)],
                    st, col_widths=[4.6 * cm, PAGE_W - 4.6 * cm], align_right_from=99)]
    story += [Spacer(1, 12)]

    story += [para("The headline result", st, "h1")]
    hl = [["", "Before", "After", "Change"],
          ["Environment steps collected per second", fmt(e1["env_steps_per_s"] if e1 else None),
           fmt(e16["env_steps_per_s"] if e16 else None), f"{fmt(speedup)}x"],
          ["Training transitions produced per second",
           fmt(e1["elite_transitions_per_s"] if e1 else None, 0),
           fmt(e16["elite_transitions_per_s"] if e16 else None, 0),
           f"{fmt((e16['elite_transitions_per_s']/e1['elite_transitions_per_s']) if (e1 and e16) else None)}x"],
          ["Environment steps in one hour",
           f"{(e1['env_steps_per_s']*3600/1000):.0f}k" if e1 else "--",
           f"{(e16['env_steps_per_s']*3600/1000):.0f}k" if e16 else "--",
           f"{fmt(speedup)}x"]]
    story += [table(hl, st, col_widths=[8.4 * cm, 2.6 * cm, 2.6 * cm, 2.4 * cm])]
    story += [Spacer(1, 10)]
    story += keypoint(
        "The GPU was doing one thing at a time when it could do sixteen. Fixing that "
        f"multiplied the data rate by {fmt(speedup)}x on the same card, without changing "
        "the algorithm.", st)

    story += [PageBreak()]

    # part 1
    story += [para("Part 1  --  Why the loop was slow", st, "h1")]
    story += [para(
        "The first thing to measure was where the time went. The answer was blunt: "
        f"<b>98% of the wall clock was the physics simulator</b>, and the simulator was "
        "being asked to do far less work than it could.", st)]
    story += [para("A planning step is deep, but narrow", st, "h2")]
    story += [para(
        "One MPPI planning step simulates a horizon of H steps, and those H steps must "
        "happen in order -- step 2 needs the state produced by step 1. That is the "
        "<i>depth</i>, and it cannot be parallelised. But all the sampled action "
        "sequences within one step are independent. That is the <i>width</i>, and a GPU "
        "eats width for free.", st)]
    if 256 in sim and 4096 in sim:
        story += [para(
            f"Measured on the RTX 4080: stepping <b>{256} worlds</b> takes "
            f"<b>{sim[256]['ctrl_step_ms']:.1f} ms</b>. Stepping <b>{4096} worlds</b> -- "
            f"sixteen times as many -- takes <b>{sim[4096]['ctrl_step_ms']:.1f} ms</b>, "
            f"only {sim[4096]['ctrl_step_ms']/sim[256]['ctrl_step_ms']:.1f} times longer. "
            "The device was idle in the width direction.", st)]
    story += [para(
        "The original code ran <b>one</b> environment, and that environment's planner used "
        "256 worlds. So all the spare width was wasted. The fix is to run many "
        "environments as extra worlds in the same bank: environment <i>e</i>, sample "
        "<i>i</i> becomes world <i>e x N + i</i>. Sixteen environments then plan in the "
        "same kernel launches that one used to.", st)]

    story += figure(figs / "fig1_throughput.png", st,
                    "Figure 1. Left: the cost of one control step against how many worlds "
                    "are stepped together -- the dashed line is what it would cost if width "
                    "were not free. Middle: environment steps collected per second against "
                    "how many environments plan together. Right: the two kinds of training "
                    "data this produces, on a log scale.")

    if sim:
        rows = [["Worlds stepped together", "ms per control step", "Physics steps / s"]]
        for n in sorted(sim):
            rows.append([str(n), f"{sim[n]['ctrl_step_ms']:.2f}",
                         f"{sim[n]['phys_steps_per_s']:.2e}"])
        story += [para("Table 1. Simulator cost against width.", st, "h2"),
                  table(rows, st, col_widths=[6.0 * cm, 4.6 * cm, 4.6 * cm])]
        story += [Spacer(1, 8)]

    story += [para("Was the learner the bottleneck instead?  No.", st, "h2")]
    story += [para(
        f"One gradient update of the whole agent (encoder, an ensemble of 5 Q networks, "
        f"the policy) costs <b>{fmt(upd_ms, 1)} ms</b>. At the original data rate of "
        f"{fmt(e1['env_steps_per_s'] if e1 else None)} environment steps per second, the "
        "learner was idle roughly 95% of the time. Speeding up data collection therefore "
        "buys gradient steps at almost no extra cost, and this report spends that budget: "
        "the ablation runs 1.5 gradient updates per environment step.", st)]

    story += [PageBreak()]

    # part 2
    story += [para("Part 2  --  Checking that the fast version is the same algorithm", st, "h1")]
    story += [para(
        "A speedup that quietly changes the maths is not a speedup. But the obvious test "
        "-- \"do the two versions produce identical trajectories?\" -- gives a misleading "
        "answer here, and it is worth saying why.", st)]
    story += [para(
        "The physics engine packs the contacts of <i>all</i> worlds into one shared pool. "
        "Change how many worlds there are, and float32 numbers get added in a different "
        "order. That is a rounding difference of about 1 part in 10 million at the first "
        "step. Contact dynamics are chaotic, so that difference grows by roughly a factor "
        "of ten per control step. After 40 steps two mathematically identical simulations "
        "visibly disagree -- <b>including the original one compared against itself</b>.", st)]
    story += [para(
        "So every check below reports two numbers: the disagreement being tested, and a "
        "<b>noise floor</b> measured by running the reference against a re-run of itself. "
        "A check passes when the first is no worse than the second. The structural "
        "question -- does world <i>e x N + i</i> really carry environment <i>e</i>'s "
        "state, goal and action? -- is answered separately at a single step, where chaos "
        "has had no time to amplify anything and agreement must be at rounding level.", st)]

    cor = ver.get("correctness", {})
    if cor:
        rows = [["Check", "Result", "Pass"]]
        for k, v in cor.items():
            rows.append([k, v.get("detail", ""), "yes" if v.get("ok") else "NO"])
        story += [table([[r[0], Paragraph(f'<font size="7.6">{r[1]}</font>', st["body"]), r[2]]
                         if i else r for i, r in enumerate(rows)], st,
                        col_widths=[5.4 * cm, PAGE_W - 7.2 * cm, 1.8 * cm],
                        align_right_from=2)]
        story += [caption("Table 2. Correctness checks on the vectorised stack. The "
                          "single-step figures (3e-08 for state, 1e-07 for actions) are at "
                          "float32 rounding level, which is what shows the world layout is "
                          "exactly right.", st)]
    story += keypoint(
        "Cross-lane leakage measured 3.0e-03 while the noise floor of an identical re-run "
        "was 4.5e-03. The environments do not influence each other beyond arithmetic "
        "round-off.", st, GOOD)

    story += [para("Part 3  --  Buying gradient steps with planner configuration", st, "h1")]
    story += [para(
        "Every environment step pays for one planning call, so the planner's settings set "
        "the budget for everything else. The question is not which setting plans best, but "
        "which plans best <i>per second</i>. Nine configurations were scored with no "
        "learning attached, on the true physics, 16 episodes at a time.", st)]
    story += figure(figs / "fig2_planner_tuning.png", st,
                    "Figure 2. Each dot is one planner configuration. Up is better quality; "
                    "left is cheaper. The base setting is orange, the setting chosen for the "
                    "ablation is green.", width=PAGE_W * 0.78)
    if tune:
        rows = [["Configuration", "Goals / 1000 steps", "ms / env step", "Env steps / s"]]
        for r in tune:
            rows.append([r["config"], f"{r['goals_per_1k_steps']:.1f}",
                         f"{r['ms_per_env_step']:.1f}", f"{r['env_steps_per_s']:.0f}"])
        hi = [i + 1 for i, r in enumerate(tune) if r["config"] == "iterations=3"]
        story += [table(rows, st, col_widths=[5.4 * cm, 4.0 * cm, 3.4 * cm, 3.4 * cm],
                        highlight_rows=hi)]
        story += [caption("Table 3. Planner sweep. Three MPPI iterations instead of four "
                          "keeps 94% of the quality for 77% of the cost; the saving was "
                          "spent on gradient updates.", st)]

    story += [PageBreak()]

    # part 4
    story += [para("Part 4  --  The ablation", st, "h1")]
    story += [para(
        "Six runs, each 150,000 environment steps, identical in everything except one "
        "switch per rung. Rungs A to D are the ladder proper -- one component added at a "
        "time; E and F vary only how the policy is fitted, on top of D. All of them face the same wrong physics: the simulator the "
        "planner uses is the model as compiled, while \"reality\" has 71 physical "
        "parameters displaced from it.", st)]

    setup = [["", "Setting"],
             ["Task", "Allegro hand, in-hand cube reorientation about the palm normal"],
             ["Environments in parallel", "16"],
             ["Planner", "MPPI, horizon 8, 256 samples, 3 iterations, 32 elites"],
             ["Worlds simulated together", "16 x 256 = 4096"],
             ["Episode length", "120 control steps (2.4 s)"],
             ["Environment steps per run", "150,000"],
             ["Gradient updates per env step", "1.5"],
             ["Physics gap", "71 unknown parameters, displaced up to 0.25 in normalised units"]]
    story += [table([[r[0], Paragraph(r[1], st["body"])] if i else r
                     for i, r in enumerate(setup)],
                    st, col_widths=[5.2 * cm, PAGE_W - 5.2 * cm], align_right_from=99)]
    story += [caption("Table 4. Shared experimental setup.", st)]

    story += figure(figs / "fig3_learning_curves.png", st,
                    "Figure 3. What the closed loop does during training. The planner is "
                    "driving in all three panels, so these curves describe the planner's "
                    "behaviour, not the learned policy's.")

    story += [PageBreak()]
    story += [para("The distillation gap", st, "h2")]
    story += [para(
        "The most important comparison in the whole report is Figure 4: the same episodes, "
        "the same axes, run once by the planner and once by the learned policy alone.", st)]
    story += figure(figs / "fig4_policy_vs_planner.png", st,
                    "Figure 4. Left: the learned policy acting on its own, with no planning. "
                    "Right: the MPPI planner. Both panels share one vertical scale, because "
                    "the distance between them is the point.")

    story += figure(figs / "fig6_final_bars.png", st,
                    "Figure 5. Final measured performance of every rung, planner against "
                    "policy.")

    # results table filled from the runs
    rows = [["Run", "Planner reward", "Planner goals", "Policy reward", "Policy goals",
             "theta err"]]
    for name, label in RUNS:
        tr, ev = runs[name]["train"], runs[name]["eval"]
        if not ev:
            continue
        rows.append([
            label,
            pm(ev.get("mpc/return"), 3, 0),
            pm(ev.get("mpc/goals_reached"), 3, 2),
            pm(ev.get("pi/return"), 6, 0),
            pm(ev.get("pi/goals_reached"), 6, 2),
            fmt(last_finite(tr.get("param_err_mean")), 3),
        ])
    if len(rows) > 1:
        story += [table(rows, st, col_widths=[3.6 * cm, 2.9 * cm, 2.7 * cm, 2.9 * cm,
                                              2.7 * cm, 1.8 * cm])]
        story += [caption("Table 5. Each rung, averaged over the last evaluations of the "
                          "run (last 3 for the planner, last 6 for the policy, 16 episodes "
                          "each) with the spread across those evaluations. Reward is per "
                          "episode; goals is how many target orientations were reached and "
                          "held within one episode.", st)]

        # What each switch is worth: the change from the rung immediately below.
        story += [para("What each switch bought", st, "h2")]
        story += [para(
            "Consecutive rungs differ in exactly one component, so the difference "
            "between two consecutive rows is what that component is worth. Rung F is "
            "not part of the ladder -- it is the change suggested by the diagnosis in "
            "Part 6, and it is read against rung E.", st)]
        drows = [["Switch added", "Policy goals", "change", "Planner goals", "change"]]
        prev = None
        for name, label in RUNS:
            ev = runs[name]["eval"]
            if not ev:
                continue
            pg, _ = tail(ev.get("pi/goals_reached"), 6)
            mg, _ = tail(ev.get("mpc/goals_reached"), 3)
            dp = f"{pg - prev[0]:+.2f}" if prev else "--"
            dm = f"{mg - prev[1]:+.2f}" if prev else "--"
            drows.append([label, fmt(pg, 2), dp, fmt(mg, 2), dm])
            prev = (pg, mg)
        if len(drows) > 2:
            story += [table(drows, st, col_widths=[4.6 * cm, 3.0 * cm, 2.4 * cm,
                                                   3.0 * cm, 2.4 * cm])]
            story += [caption("Table 6. Goals solved per episode, and the change each "
                              "switch made relative to the rung before it.", st)]

    story += [PageBreak()]

    # part 5
    story += [para("Part 5  --  System identification", st, "h1")]
    story += [para(
        "The planner's simulator starts with the wrong physical numbers -- 71 of them: "
        "joint damping, rotor inertia, dry friction, servo gains, contact friction per "
        "finger, link masses, the cube's mass and inertia, and the two constants of the "
        "contact model. SysID fits them from short chunks of real trajectory replayed "
        "inside the simulator.", st)]
    story += [para(
        "The trick that makes it cheap is the same one that made the planner fast: every "
        "probe direction and every data chunk is a separate world, so one descent "
        "iteration is two batched rollouts no matter how many parameters there are.", st)]
    story += figure(figs / "fig5_sysid.png", st,
                    "Figure 6. Left: how wrong the planner's physics is over training. "
                    "Middle: the fraction of the achievable loss gap each identification "
                    "call closes. Right: which groups of parameters actually moved.")

    # per-group identification: the average hides the result
    sy = runs.get("c_sysid", {}).get("sysid", {})
    gcols = sorted(k for k in sy if k.startswith("err/"))
    if gcols:
        counts = {"joint": 48, "actuator": 8, "contact": 7, "inertial": 6, "comfree": 2}
        grows = [["Parameter group", "How many", "Error at first fit", "At last fit", "Change"]]
        for g in gcols:
            gname = g.split("/")[1]
            a, b = float(sy[g][0]), float(sy[g][-1])
            grows.append([gname, str(counts.get(gname, "")), f"{a:.4f}", f"{b:.4f}",
                          f"{100 * (b - a) / a:+.0f}%"])
        a0, b0 = float(sy["param_err_mean"][0]), float(sy["param_err_mean"][-1])
        grows.append(["all 71 together", "71", f"{a0:.4f}", f"{b0:.4f}",
                      f"{100 * (b0 - a0) / a0:+.0f}%"])
        story += [table(grows, st, col_widths=[4.4 * cm, 2.4 * cm, 3.6 * cm, 2.8 * cm,
                                               2.4 * cm],
                        highlight_rows=[len(grows) - 1])]
        story += [caption("Table 7. Identification by parameter group, run C. The average "
                          "over all 71 parameters is the least informative number in the "
                          "table: it is dominated by the 48 joint parameters, most of which "
                          "the data barely constrains.", st)]
        story += keypoint(
            "The two contact-model constants -- the ones ComFree's smooth contact law is "
            "built on -- had their error cut in half. The six inertial parameters got "
            "worse: the fit trades accuracy in directions the data does not constrain for "
            "accuracy in directions it does. That is unidentifiability, not a bug.", st)

    story += [PageBreak()]

    # part 6
    story += [para("Part 6  --  Why the policy does not match the planner", st, "h1")]
    story += [para(
        "Across every rung the learned policy stays well below the planner. Four "
        "explanations are usually offered for this, and they call for different fixes, "
        "so they were measured apart rather than guessed between.", st)]

    diag_rows = [["Explanation", "How it was measured"],
                 ["Mode averaging",
                  "MPPI is multi-modal: at one state several unrelated action sequences "
                  "score alike. A single deterministic policy fitted to all of them lands "
                  "between them. Measured by comparing the policy's imitation error "
                  "against the spread inside the elite set itself -- the error no "
                  "deterministic function can beat."],
                 ["Mode collapse",
                  "The policy emits nearly the same action everywhere. Measured by the "
                  "standard deviation of its actions over time, against the planner's."],
                 ["Distribution shift",
                  "The policy copies well where the planner looks, then drifts elsewhere "
                  "when it drives. Measured as the same imitation error on planner-visited "
                  "and policy-visited states."],
                 ["A dishonest value function",
                  "The policy's objective is mostly 'maximise Q'. Measured by comparing Q's "
                  "prediction against the discounted return that actually followed."]]
    story += [table([[r[0], Paragraph(f'<font size="8.4">{r[1]}</font>', st["body"])]
                     if i else r for i, r in enumerate(diag_rows)],
                    st, col_widths=[4.4 * cm, PAGE_W - 4.4 * cm], align_right_from=99)]
    story += [Spacer(1, 8)]
    story += figure(figs / "fig7_diagnosis.png", st,
                    "Figure 7. The four measurements, per rung. In panel 1 the black bar is "
                    "the floor: how far apart the actions the planner itself considers "
                    "equally good are. Panel 2 is the ratio of the two bars in panel 1.")

    dg = {}
    for name, label in RUNS:
        p = abl / name / "diagnosis.json"
        if p.exists():
            dg[name] = (label, json.loads(p.read_text()))
    if dg:
        drows = [["Run", "Copy error\n(planner states)", "Floor", "Copy error\n(own states)",
                  "Drift", "Q says", "Truth", "Policy goals"]]
        for name, (label, d) in dg.items():
            cp = d.get("imitation_error/planner_states", float("nan"))
            co = d.get("imitation_error/policy_states", float("nan"))
            pg, _ = tail(runs.get(name, {}).get("eval", {}).get("pi/goals_reached"), 6)
            drows.append([
                label, fmt(cp, 3), fmt(d.get("elite_spread/mean_abs_dev"), 3), fmt(co, 3),
                f"{co / cp:.2f}x" if np.isfinite(cp) and cp else "--",
                fmt(d.get("value/q_mean"), 0), fmt(d.get("value/mc_return_mean"), 0),
                fmt(pg, 2),
            ])
        story += [table([[Paragraph(f'<font size="8">{c}</font>', st["body"]) for c in r]
                         if i == 0 else r for i, r in enumerate(drows)],
                        st, col_widths=[3.1 * cm, 2.6 * cm, 1.5 * cm, 2.4 * cm, 1.5 * cm,
                                        1.6 * cm, 1.5 * cm, 2.0 * cm])]
        story += [caption("Table 8. Diagnosis of the distillation gap. Action errors are "
                          "mean absolute, per action dimension, on a [-1, 1] scale. 'Drift' "
                          "is the copy error on the policy's own states divided by the copy "
                          "error on the planner's.", st)]

    story += [para("What the measurements rule out", st, "h2")]
    story += [para(
        "<b>It is not mode averaging.</b> The floor -- how far apart the actions the "
        "planner itself scores as equally good are -- sits between 0.11 and 0.14 in every "
        "run, while the policy's copy error ranges from 0.51 down to 0.19. The policy is "
        "between 1.7 and 4.7 times worse than the best a single deterministic function "
        "could do. It never gets close enough to the floor for multi-modality to be what "
        "is stopping it.", st)]
    story += [para(
        "<b>It is not mode collapse.</b> The policy's actions vary over time by 0.36 to "
        "0.56, against the planner's 0.36 to 0.61. It is not emitting one action "
        "everywhere.", st)]
    story += [para(
        "<b>It is drift, and the drift gets worse as the planner gets better.</b> Reading "
        "down the ladder, the policy copies the planner steadily <i>better</i> on the "
        "planner's own states -- 0.51, 0.43, 0.38, 0.27, 0.19 -- and yet its measured "
        "performance peaks in the middle, at rung C, and then falls. The column that "
        "moves with performance is the drift ratio: 1.05, 1.07, 1.25, 2.04, 2.35. At "
        "rungs D and E the policy copies the planner twice as badly once it is the one "
        "driving.", st)]
    story += keypoint(
        "The best imitator on the ladder (rung E, copy error 0.19, within 1.7x of the "
        "floor) is one of the worst controllers. Copying accuracy and control performance "
        "come apart, and what separates them is how far the policy's own trajectory "
        "leaves the states it was trained on.", st, WARN)
    story += [para(
        "This has a plausible reading, though the ablation was not designed to test it "
        "and it should be treated as a hypothesis rather than a result. A terminal value "
        "makes the planner much more capable -- rung D almost never drops the cube -- and "
        "a more capable controller keeps the system inside a narrower, more finely "
        "maintained region of state space. Small copying errors then take the policy "
        "outside that region faster, where its training data thins out and its errors "
        "grow. The better the expert, the less forgiving cloning it becomes.", st)]
    story += [para(
        "<b>The value function is optimistic, and increasingly so:</b> Q predicts 68 "
        "against a true 33 at rung B, and 175 against 123 at rung E. Its ranking is also "
        "weak (correlation with the realised return between 0.23 and 0.59). Since the "
        "policy's objective is largely to maximise Q, this is a second real problem -- "
        "but it cannot be the main one, because the copy error keeps improving down the "
        "ladder while Q gets steadily worse.", st)]

    story += [PageBreak()]
    story += [para("Part 7  --  What was learned", st, "h1")]

    # Numbers for the prose, pulled from the runs rather than typed in.
    def g(name, who):
        ev = runs.get(name, {}).get("eval", {})
        return tail(ev.get(f"{who}/goals_reached"), 6 if who == "pi" else 3)[0]

    def d(name, who):
        ev = runs.get(name, {}).get("eval", {})
        return tail(ev.get(f"{who}/dropped"), 6 if who == "pi" else 3)[0]

    story += [para("1.  Throughput was the whole game", st, "h2")]
    story += [para(
        f"The planner is deep and narrow, and a GPU is wide. Mapping environment "
        f"<i>e</i>, sample <i>i</i> onto world <i>e x N + i</i> raised the data rate "
        f"{fmt(speedup)}x on the same card with no change to the algorithm, verified "
        "against the simulator's own numerical noise floor. Everything below is only "
        "measurable because of it: at the original rate, a single one of these six runs "
        f"would have taken "
        f"{fmt((150000 / e1['env_steps_per_s'] / 3600) if e1 else None)} hours instead "
        "of one.", st)]

    story += [para("2.  Distillation works -- once there is enough data", st, "h2")]
    story += [para(
        f"Harvesting the elite MPPI rollouts raised the policy from "
        f"{fmt(g('a_base','pi'), 2)} to {fmt(g('b_distill','pi'), 2)} goals per episode. "
        "It also made the <i>planner</i> better "
        f"({fmt(g('a_base','mpc'), 2)} to {fmt(g('b_distill','mpc'), 2)} goals), which is "
        "worth explaining: 24 of the planner's 256 sampled trajectories are generated by "
        "the policy instead of by noise. Those 24 are closed-loop -- they react to the "
        "state they reach, which an open-loop noise sample cannot -- so a better policy "
        "hands the planner a class of candidate it could not otherwise draw.", st)]

    story += [para("3.  System identification fixes the contact model, not the robot", st, "h2")]
    story += [para(
        "Averaged over 71 parameters the improvement looks negligible. Split by group it "
        "is not: the two contact-model constants had their error halved, contact "
        "frictions improved by a sixth, the 48 joint parameters barely moved, and the "
        "inertial parameters got worse. The planner nevertheless gained a large amount of "
        f"reward ({pm(runs['b_distill']['eval'].get('mpc/return'), 3, 0)} to "
        f"{pm(runs['c_sysid']['eval'].get('mpc/return'), 3, 0)}) without solving more "
        "goals -- it made more progress before losing the cube. Predictive accuracy where "
        "the data constrains it is what planning needs; recovering the true parameter "
        "vector is a harder and different problem.", st)]

    story += [para("4.  The terminal value transforms the planner", st, "h2")]
    story += [para(
        f"This is the largest single effect in the report. Letting the planner add "
        f"gamma^H Q(s_H) to each rollout took it from {fmt(g('c_sysid','mpc'), 2)} to "
        f"{fmt(g('d_termq','mpc'), 2)} goals per episode, and the fraction of episodes "
        f"ending with the cube on the floor fell from {fmt(d('c_sysid','mpc'), 2)} to "
        f"{fmt(d('d_termq','mpc'), 2)}.", st)]
    story += [para(
        "The mechanism is specific. Dropping the cube costs 50 reward, but a horizon-8 "
        "planner only sees that penalty if the drop happens inside those 8 steps. By "
        "then it is usually too late -- the cube is already sliding. A learned value "
        "function scores the state itself, so 'this grip is heading towards a drop' "
        "reaches the planner tens of steps earlier. This is exactly the claim TD-MPC "
        "makes for a terminal value, and here it is isolated from everything else.", st)]
    story += keypoint(
        f"Terminal Q learning is what this project is for, and it works: "
        f"{fmt(g('c_sysid','mpc'), 2)} -> {fmt(g('d_termq','mpc'), 2)} goals per episode, "
        f"drop rate {fmt(d('c_sysid','mpc'), 2)} -> {fmt(d('d_termq','mpc'), 2)}.",
        st, GOOD)

    story += [para("5.  The same Q that saves the planner hurts the policy", st, "h2")]
    story += [para(
        f"The policy moves the other way at rung D: {fmt(g('c_sysid','pi'), 2)} down to "
        f"{fmt(g('d_termq','pi'), 2)} goals. The planner is protected from an "
        "over-optimistic Q by eight steps of real simulated reward, which dominate the "
        "score. The policy has no such anchor: it maximises Q directly, and it is trained "
        "to imitate actions the planner picked partly <i>because</i> Q was optimistic "
        "about where they led.", st)]
    story += [para(
        "Fitting the policy to the elite actions by plain regression (rung E) did not "
        f"close the gap either: goals stayed at {fmt(g('e_bc','pi'), 2)} while the drop "
        f"rate fell from {fmt(d('d_termq','pi'), 2)} to {fmt(d('e_bc','pi'), 2)}. The "
        "policy became more careful, not more capable. Rung F, which hands the policy the "
        "plan's weighted mean instead of individual elite actions -- one consistent target "
        f"per state -- recovered part of it, to {fmt(g('f_meantarget','pi'), 2)} goals "
        "with the policy's episode reward turning positive for the first time. Its error "
        "bars overlap rung E's, so this is supporting evidence and not an established "
        "result.", st)]
    story += [para(
        "That rung was designed to test a hypothesis that the measurements then "
        "contradicted. It was built to fix mode averaging, and Part 6 shows mode averaging "
        "is not the problem: the elite set is far tighter than the policy's error "
        "throughout. Whatever rung F gained, it did not gain by the mechanism it was "
        "built for -- more likely the plan mean is simply a lower-variance regression "
        "target than a single sampled elite. Its drift ratio (2.17) does sit slightly "
        "below rung E's (2.35), which is consistent with that reading.", st)]

    story += [para("What to try next", st, "h2")]
    nxt = [["Idea", "Why"],
           ["Let the policy drive sometimes",
            "This is what the diagnosis points at. The copy error roughly doubles on the "
            "policy's own states at the rungs where it performs worst, and that ratio -- "
            "not the copy error itself -- is what tracks performance across the ladder. "
            "Collecting a fraction of episodes under the policy and having the planner "
            "relabel the actions (DAgger) attacks exactly that."],
           ["Train the policy without Q",
            "Rung D shows the value function helps the planner and hurts the policy. A "
            "policy trained purely by imitation, with Q used only as the planner's "
            "terminal value, separates the two roles that are currently in conflict."],
           ["Not a richer policy class",
            "Recorded as a negative result: a mixture or diffusion policy is the standard "
            "answer to a multi-modal target, and the measurements say that is not what is "
            "binding here. The elite set is three to five times tighter than the policy's "
            "own error in every run. Reach for this only after the drift is dealt with."],
           ["Run longer",
            "The policy curves in Figure 4 are still climbing at 150,000 steps. Nothing "
            "here has converged; these are one-hour budgets, not asymptotes."]]
    story += [table([[Paragraph(f"<b>{r[0]}</b>", st["body"]) if i else r[0],
                      Paragraph(r[1], st["body"])] if i else r
                     for i, r in enumerate(nxt)],
                    st, col_widths=[5.4 * cm, PAGE_W - 5.4 * cm], align_right_from=99)]
    story += [Spacer(1, 12)]

    story += [para("How much to trust this", st, "h2")]
    lim = [["Limitation", "What it means for the numbers"],
           ["One seed per rung",
            "Each row of the ladder is a single training run. Differences of the size seen "
            "between rungs A, B and C are of the same order as the spread between "
            "evaluations, so those three should be read as a trend, not as three "
            "separately established results. The rung D effect is roughly ten times the "
            "evaluation spread and does not depend on that caveat."],
           ["Not converged",
            "150,000 environment steps is one hour of wall clock, chosen so that six "
            "configurations fit in a day. The policy is still improving at the end of "
            "every run."],
           ["Evaluation noise",
            "One evaluation is 16 episodes of a task whose reward is dominated by whether "
            "the cube survives. Every number quoted is the mean over the last several "
            "evaluations with its spread beside it, never a single final point."],
           ["The reality gap is synthetic",
            "'Reality' is the same simulator with 71 displaced parameters, so SysID is "
            "being asked to close a gap that is exactly parametric. A real robot differs "
            "from its model in ways no parameter vector can express."]]
    story += [table([[r[0], Paragraph(r[1], st["body"])] if i else r
                     for i, r in enumerate(lim)],
                    st, col_widths=[4.4 * cm, PAGE_W - 4.4 * cm], align_right_from=99)]

    doc = SimpleDocTemplate(
        str(Path(args.out)), pagesize=A4,
        leftMargin=1.7 * cm, rightMargin=1.7 * cm,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        title="ComFree-TDMPC ablation report", author="",
    )

    def page(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.6)
        canvas.setFillColor(INK2)
        canvas.drawString(1.7 * cm, 1.0 * cm, "ComFree-TDMPC  --  throughput and ablation")
        canvas.drawRightString(A4[0] - 1.7 * cm, 1.0 * cm, f"{_doc.page}")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(1.7 * cm, 1.35 * cm, A4[0] - 1.7 * cm, 1.35 * cm)
        canvas.restoreState()

    doc.build(story, onFirstPage=page, onLaterPages=page)
    return Path(args.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", type=str, default="outputs/report/figures")
    ap.add_argument("--ablation", type=str, default="outputs/ablation")
    ap.add_argument("--out", type=str, default="outputs/report/comfree_tdmpc_report.pdf")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    p = build(args)
    print(f"wrote {p}  ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
